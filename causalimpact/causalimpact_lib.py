# Copyright 2019-2023 The TFP CausalImpact Authors
# Copyright 2014 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TFP-based CausalImpact implementation via fit_causalimpact."""

import dataclasses
import logging
import math
from typing import Dict, List, Optional, Tuple, Union

from causalimpact import posterior_processing
import causalimpact.data as cid
from causalimpact.indices import InputDateType
from causalimpact.indices import OutputDateType
from causalimpact.indices import OutputPeriodType

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.distributions import MultivariateNormalPrecisionFactorLinearOperator
from tensorflow_probability.python.experimental.sts_gibbs import gibbs_sampler
from tensorflow_probability.python.internal import prefer_static as ps

tfb = tfp.bijectors
tfd = tfp.distributions

TensorLike = tf.types.experimental.TensorLike
_SeedType = Union[int, Tuple[int, int], TensorLike]


@dataclasses.dataclass
class CausalImpactPosteriorSamples:
  """Results for samples of the latents of CausalImpact."""
  # Shape is [num_samples].
  observation_noise_scale: tf.Tensor
  # Shape is [num_samples].
  level_scale: Optional[tf.Tensor]
  # Shape is [num_samples].
  level: Optional[tf.Tensor]
  # Shape is [num_samples, num_covariates + 1 (for intercept)].
  weights: Optional[tf.Tensor]
  # Shape is [num_samples, num_seasonal_effects]
  seasonal_drift_scales: Optional[tf.Tensor]
  # Shape is [num_samples, num_seasonal_effects]
  seasonal_levels: Optional[tf.Tensor]


@dataclasses.dataclass
class CausalImpactAnalysis:
  """Results of a CausalImpact analysis.

  Holds both the dataframe outputs of `fit_causalimpact`, as well as the full
  posterior of draws from the structural time series model that was fit. See
  `build_model_for_gibbs_fitting` in the module
  `tensorflow_probability.experimental.sts_gibbs.gibbs_sampler` for details of
  the model parameters.

  Attributes:
    series: DataFrame describing the posterior of the causal impact of the
      intervention at each step. This contains an index with the same
      range as the input data (i.e. not just constrained to
      [pre-period start, post-period end]) with columns:

        observed: The observed value.
        posterior_mean: The mean of the predicted value.
        posterior_lower: The lower bound of the (equal-tailed) credible interval
          for the counterfactual forecast of the observed value.
        posterior_upper: The upper bound of the (equal-tailed) credible interval
          for the counterfactual forecast of the observed value.
        point_effects_mean: The posterior mean of the difference between the
          observed and predicted value.
        point_effects_lower: The lower bound of the `(1-alpha)`-percentile
          (equal-tailed) credible interval for the difference between the
          observed and the predicted value.
        point_effects_upper: The upper bound of the `(1-alpha)`-percentile
          (equal-tailed) credible interval for the difference between the
          observed and the predicted value.
        cumulative_effects_mean: Starting from the beginning of the post-period,
          the posterior mean of the cumulative difference between the
          observed and predicted value.
        cumulative_effects_lower: The lower bound of the `(1-alpha)`-percentile
          (equal-tailed) credible interval for the cumulative difference between
          the observed and the predicted value.
        cumulative_effects_upper: The upper bound of the `(1-alpha)`-percentile
          (equal-tailed) credible interval for the cumulative difference between
          the observed and the predicted value.

      Some values may be NaN.
    summary: DataFrame that is a compact representation of `series` that
      summarizes the impact in the post-period. The two rows - labelled
      'average' and 'cumulative' - are computed from the posterior predictive
      samples, and for the first nine columns, the `cumulative` will be a
      constant multiple of the `average`. The columns are:

        actual: The observed value over the post-period.
        predicted: The predicted value over the post-period.
        predicted_lower: Lower bound for the `(1-alpha)`-percentile
          credible interval of the predicted value.
        predicted_upper: Upper bound for the `(1-alpha)`-percentile
          (equal-tailed) credible interval of the predicted value.
        predicted_sd: Standard deviation of the predicted value.
        abs_effect: The posterior mean of the difference between the
          observed and predicted value.
        abs_effect_lower: The lower bound of the `(1-alpha)`-percentile
          (equal-tailed) credible interval for the difference between the
          observed and the predicted value.
        abs_effect_upper: The upper bound of the `(1-alpha)`-percentile
          (equal-tailed) credible interval for the difference between the
          observed and the predicted value.
        abs_effect_sd: Standard deviation of the difference between the
          aggregation of the actual value and the predicted value.
        rel_effect: Posterior mean of the relative effect
          (observed / predicted - 1). This is the same for both average and
          cumulative.
        rel_effect_lower: The lower bound of the `(1-alpha)`-percentile
          (equal-tailed) credible interval of the relative effect.
        rel_effect_upper: The upper bound of the `(1-alpha)`-percentile
          (equal-tailed) credible interval of the relative effect.
        rel_effect_sd: The standard deviation of the relative effect.
        p_value: One-sided p-value that there was a cumulative effect.

    posterior_samples: NamedTuple of Tensorlikes containing samples of latent
      variables. Useful for understanding what the model learned without even
      considering predictions.
  """
  series: pd.DataFrame
  summary: pd.DataFrame
  # Samples of latent variables of the model. Note that in CausalImpact.R,
  # impact$posterior_samples refers to predictive posterior samples, instead
  # of samples of the latents.
  posterior_samples: CausalImpactPosteriorSamples


@dataclasses.dataclass
class DataOptions:
  """Options for working with the input data.

  Attributes:
    outcome_column: String giving the name of the outcome column in `data`. If
      not specified, the first column in `data` is used.
    standardize_data: If covariates and output should be standardized.
    dtype: The dtype to use throughout computation.
  """
  outcome_column: Optional[str] = None
  standardize_data: bool = True
  dtype: tf.dtypes.DType = tf.float32


@dataclasses.dataclass(frozen=True)
class Seasons:
  """Options for modeling a single seasonal effect.

  For instance, modeling day of the week.

  Attributes:
    num_seasons: Number of seasons.
    num_steps_per_season: Number of steps in each season. This may be either an
      int, in which case all seasons have the same length, or a tuple of ints of
      shape `[num_seasons]`, in which seasons have different length, but remain
      constant around different cycles, or a tuple of tuples of ints of shape
      `[num_cycles, num_seasons]`, in which num_steps_per_season for each season
      also varies in different cycle (e.g., a 4 years cycle with leap day).
      Using [num_cycles] allows having the timeseries start in the middle of a
      season.
  """
  num_seasons: int
  num_steps_per_season: Union[int, Tuple[int], Tuple[Tuple[int]]] = 1


@dataclasses.dataclass
class ModelOptions:
  """Options used for the model that is being fit.

  Attributes:
    prior_level_sd: float: Prior standard deviation of the Gaussian random walk
      of the local level model. Expressed in terms of data standard deviations.
      Defaults to 0.01, a typical choice for well-behaved and stable datasets
      with low residual volatility. When in doubt, a safer option is to use 0.1,
      as validated on synthetic data, although this may sometimes give rise to
      unrealistically wide prediction intervals.
    seasons: This supports a list of Seasons, for modeling
      multiple seasons. For instance, for hourly data, there could be
      both an hour-of-the-day (num_seasons=24, num_steps_per_season=1), and
      day-of-the-week effect (num_seasons=7, num_steps_per_season=24) by passing
      in two SeasonalOptions. This is different from a single SeasonalOption
      that would model hour-of-the-day-of-the-week effect (num_seasons=7*24,
      num_steps_per_season=1).
  """
  prior_level_sd: float = 0.01
  seasons: List[Seasons] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class InferenceOptions:
  """Options for how to do inference for the model.

  Attributes:
    num_results: integer number of posterior samples to draw.
    num_warmup_steps: integer number of steps to take before starting to collect
      results. If not set, then 1/9th of num_results will be used.
  """
  num_results: int = 900
  num_warmup_steps: Optional[int] = None

  def __post_init__(self):
    if self.num_warmup_steps is None:
      self.num_warmup_steps = math.ceil(self.num_results / 9)


def fit_causalimpact(data: pd.DataFrame,
                     pre_period: Tuple[InputDateType, InputDateType],
                     post_period: Tuple[InputDateType, InputDateType],
                     alpha: float = 0.05,
                     seed: Optional[_SeedType] = None,
                     data_options: Optional[DataOptions] = None,
                     model_options: Optional[ModelOptions] = None,
                     inference_options: Optional[InferenceOptions] = None,
                     **kwargs) -> CausalImpactAnalysis:
  """Fit a CausalImpact model on provided data.

  Args:
    data: Pandas `DataFrame` containing an outcome time series and optional
      features time series. If a datetime index is used, it is assumed to be of
      a fixed period. Similarly, an integer index assumes all points are
      adjacent (e.g. an index of [2,3,4,6] will assume [4,6] are adjacent for
      training and prediction.
    pre_period: Tuple of datetimes, indices or strings (convertible via
      pd.to_datetime) giving pre-period start/end.
    post_period: Tuple of datetimes, indices or strings (convertible via
      pd.to_datetime) giving post-period start/end.
    alpha: float, used to get confidence level for predictions.
    seed: PRNG seed; see `tensorflow_probability.random.sanitize_seed` for
      details.
    data_options: See DataOptions.
    model_options: See ModelOptions.
    inference_options: See InferenceOptions.
    **kwargs: Only for experimental options. Unsupported keys will raise an
      exception.

  Returns:
    A CausalImpactAnalysis instance summarizing the effect of the intervention.
  """
  # Suppress verbose TensorFlow WARNING and INFO messages, which are not useful
  # for TFP CausalImpact users.
  tf_log_level = tf.get_logger().level
  tf.get_logger().setLevel(logging.ERROR)
  try:
    data_options = data_options if data_options is not None else DataOptions()
    model_options = (model_options if model_options is not None
                     else ModelOptions())
    inference_options = (inference_options if inference_options is not None
                         else InferenceOptions())

    # WARNING: These are implementation details, and have no guarantees of
    # continuing to work or be respected.
    experimental_model = kwargs.pop("experimental_model", None)
    experimental_tf_function_cache_key_addition = kwargs.pop(
        "experimental_tf_function_cache_key_addition", 0)
    if kwargs:
      raise TypeError(f"Received unknown {kwargs=}")

    ci_data = cid.CausalImpactData(
        data=data,
        pre_period=pre_period,
        post_period=post_period,
        outcome_column=data_options.outcome_column,
        standardize_data=data_options.standardize_data,
        dtype=data_options.dtype)
    posterior_samples, posterior_means, posterior_trajectories = _train_causalimpact_sts(
        ci_data=ci_data,
        prior_level_sd=model_options.prior_level_sd,
        seed=seed,
        num_results=inference_options.num_results,
        num_warmup_steps=inference_options.num_warmup_steps,
        model=experimental_model,
        dtype=data_options.dtype,
        seasons=model_options.seasons,
        experimental_tf_function_cache_key_addition=experimental_tf_function_cache_key_addition
    )
    series, summary = _compute_impact(
        posterior_means=posterior_means,
        posterior_trajectories=posterior_trajectories,
        ci_data=ci_data,
        alpha=alpha)

    if posterior_samples.seasonal_levels.shape[-1] > 0:
      # If we have k seasonal effects, with num_seasons[0],...,num_seasons[k-1]
      # distinct seasons respectively, then each seasonal effect's latent state
      # has dimension num_seasons[0]-1, ..., num_seasons[k-1]-1 and the shape
      # of posterior_samples.seasonal_levels is
      #   batch_shape [timeseries_length, total_seasonal_latent_dim]
      # where
      #   total_seasonal_latent_dim = num_seasons[0]-1, ..., num_seasons[k-1]-1.
      #
      # And at each timestep, each seasonal effect's contribution to the
      # observed value is the 0-th element of its latent state.  Here we extract
      # these 0-th elements in order to return  the contribution of each
      # seasonal effect at each time step.
      seasonal_levels = []
      index = 0
      for season in model_options.seasons:
        seasonal_levels.append(posterior_samples.seasonal_levels[..., index])
        index += season.num_seasons - 1
      seasonal_levels = tf.stack(seasonal_levels, axis=-1)
    else:
      # If there are no seasonal effects, then we can just use
      # posterior_samples.seasonal_levels, which has the correct shape:
      #     batch_shape + [num_samples, timeseries_length, 0]
      seasonal_levels = posterior_samples.seasonal_levels

    # Translate posterior samples to a CausalImpact-specific object,
    # rather than exposing GibbsSamplerState.
    ci_posterior_samples = CausalImpactPosteriorSamples(
        observation_noise_scale=posterior_samples.observation_noise_scale,
        level_scale=posterior_samples.level_scale,
        level=posterior_samples.level,
        weights=(posterior_samples.weights
                 if posterior_samples.weights.shape[1] > 0 else None),
        seasonal_drift_scales=(
            posterior_samples.seasonal_drift_scales
            if posterior_samples.seasonal_drift_scales.shape[-1] > 0 else None),
        seasonal_levels=seasonal_levels
    )
    return CausalImpactAnalysis(series, summary, ci_posterior_samples)
  finally:
    tf.get_logger().setLevel(tf_log_level)


# Always use graph mode: eager mode is very, very slow. The non-compiled version
# is used with the dynamic Cholesky decompositions, which use a dynamic shape,
# and frustrates the compilation.
@tf.function(autograph=False, jit_compile=False)
def _run_gibbs_sampler(
    sts_model: Optional[tfp.sts.StructuralTimeSeries], outcome_ts: TensorLike,
    outcome_sd: TensorLike, design_matrix: Optional[TensorLike],
    num_results: int, num_warmup_steps: int,
    observation_noise_scale: TensorLike, level_scale: TensorLike,
    seasonal_drift_scales: TensorLike, weights: TensorLike, level: TensorLike,
    slope: TensorLike, seed: TensorLike, dtype, seasons: List[Seasons],
    experimental_tf_function_cache_key_addition: int):  # pylint: disable=unused-argument
  """Fits parameters of the model using Gibbs sampling."""
  if not sts_model:
    sts_model = _build_default_gibbs_model(
        design_matrix=design_matrix,
        outcome_ts=outcome_ts,
        level_scale=level_scale,
        outcome_sd=outcome_sd,
        dtype=dtype,
        seasons=seasons)

  sample_seed, forecast_seed = tfp.random.split_seed(seed)
  posterior_samples = gibbs_sampler.fit_with_gibbs_sampling(
      sts_model,
      outcome_ts,
      num_results=num_results,
      num_warmup_steps=num_warmup_steps,
      initial_state=gibbs_sampler.GibbsSamplerState(
          observation_noise_scale=observation_noise_scale,
          level_scale=level_scale,
          # Model has no slope component.
          slope_scale=tf.zeros([], dtype=dtype),
          weights=weights,
          level=level,
          slope=slope,
          seed=None,
          seasonal_drift_scales=seasonal_drift_scales,
          seasonal_levels=tf.zeros(
              shape=gibbs_sampler.get_seasonal_latents_shape(
                  outcome_ts.time_series, sts_model),
              dtype=dtype)),
      # TODO(colcarroll,jburnim): Move to commented-on module constant.
      default_pseudo_observations=tf.ones([], dtype=dtype) * 0.01,
      seed=sample_seed,
      experimental_use_dynamic_cholesky=True,
      experimental_use_weight_adjustment=True)

  posterior_means, posterior_trajectories = (
      _get_posterior_means_and_trajectories(
          sts_model=sts_model,
          posterior_samples=posterior_samples,
          seed=forecast_seed))
  return posterior_samples, posterior_means, posterior_trajectories


def _build_default_gibbs_model(
    design_matrix: Optional[tf.Tensor],
    outcome_ts: tfp.sts.MaskedTimeSeries,
    level_scale: tf.Tensor,
    outcome_sd: tf.Tensor,
    dtype,
    seasons: List[Seasons],
):
  """A method to build the default STS model.

  The default model is a local level model with observation noise, plus optional
  sparse regression using observed covariates, if given.

  Args:
    design_matrix: Optional Tensor of [timesteps, features] with the covariates.
    outcome_ts: An instance of tfp.sts.MaskedTimeSeries.
    level_scale: tf.Tensor - Initial scale for the local level.
    outcome_sd: Standard deviation of non-nan values in the `observed_ts`. Used
      for scaling parameter defaults.
    dtype: Desired dtype for Gibbs model.
    seasons: An interable of seasonal options for seasonal components to add
      to the model.

  Returns:
    A tfp.sts.StructuralTimeSeries instance.
  """
  local_level_prior_sample_size = tf.constant(32., dtype=dtype)

  level_concentration = tf.cast(local_level_prior_sample_size / 2., dtype=dtype)
  level_variance_prior_scale = level_scale * level_scale * (
      local_level_prior_sample_size / 2.)

  level_variance_prior = tfd.InverseGamma(
      concentration=level_concentration, scale=level_variance_prior_scale)
  level_variance_prior.upper_bound = outcome_sd

  if design_matrix is not None:
    observation_noise_variance_prior = tfd.InverseGamma(
        concentration=tf.constant(25., dtype=dtype),
        scale=tf.math.square(outcome_sd) * tf.constant(5., dtype=dtype))
  else:
    observation_noise_variance_prior = tfd.InverseGamma(
        concentration=tf.constant(0.005, dtype=dtype),
        scale=tf.math.square(outcome_sd) * tf.constant(0.005, dtype=dtype))
  observation_noise_variance_prior.upper_bound = outcome_sd * tf.constant(
      1.2, dtype=dtype)

  if design_matrix is not None:
    design_shape = ps.shape(design_matrix)
    num_outputs = design_shape[-2]
    num_dimensions = design_shape[-1]
    sparse_weights_nonzero_prob = tf.minimum(
        tf.constant(1., dtype=dtype), 3. / num_dimensions)
    x_transpose_x = tf.matmul(design_matrix, design_matrix, transpose_a=True)
    weights_prior_precision = 0.01 * tf.linalg.set_diag(
        0.5 * x_transpose_x, tf.linalg.diag_part(x_transpose_x)) / num_outputs
    # TODO(colcarroll): Remove this cholesky - it is used to instantiate the
    # MVNPFLO below, but later code only uses the precision.
    precision_factor = tf.linalg.cholesky(weights_prior_precision)

    # Note that this prior uses the entire design matrix -- not just the
    # pre-period -- which "cheats" by using future data.
    weights_prior = MultivariateNormalPrecisionFactorLinearOperator(
        precision_factor=tf.linalg.LinearOperatorFullMatrix(precision_factor),
        precision=tf.linalg.LinearOperatorFullMatrix(weights_prior_precision))
  else:
    sparse_weights_nonzero_prob = None
    weights_prior = None

  initial_level_prior = tfd.Normal(
      loc=tf.cast(outcome_ts.time_series[..., 0], dtype=dtype),
      scale=outcome_sd)

  seasonal_components = []
  seasonal_variance_prior = tfd.InverseGamma(
      concentration=0.005, scale=5e-7 * tf.square(outcome_sd))
  seasonal_variance_prior.upper_bound = outcome_sd
  for seasonal_options in seasons:
    seasonal_components.append(
        tfp.sts.Seasonal(
            num_seasons=seasonal_options.num_seasons,
            num_steps_per_season=np.array(
                seasonal_options.num_steps_per_season),
            allow_drift=True,
            constrain_mean_effect_to_zero=True,
            # TODO(colcarroll,jburnim): If there are multiple seasonal effects,
            # should the prior for the draft scale or initial effect be
            # scaled down?
            drift_scale_prior=tfd.TransformedDistribution(
                bijector=tfb.Invert(tfb.Square()),
                distribution=seasonal_variance_prior),
            initial_effect_prior=tfd.Normal(loc=0., scale=outcome_sd)))

  return gibbs_sampler.build_model_for_gibbs_fitting(
      outcome_ts,
      design_matrix=design_matrix,
      weights_prior=weights_prior,
      level_variance_prior=level_variance_prior,
      slope_variance_prior=None,
      observation_noise_variance_prior=observation_noise_variance_prior,
      initial_level_prior=initial_level_prior,
      sparse_weights_nonzero_prob=sparse_weights_nonzero_prob,
      seasonal_components=seasonal_components)


def _train_causalimpact_sts(
    *,
    ci_data: cid.CausalImpactData,
    prior_level_sd,
    seed: _SeedType,
    num_results: int,
    num_warmup_steps: int,
    model: Optional[tfp.sts.StructuralTimeSeries] = None,
    dtype,
    seasons: List[Seasons],
    experimental_tf_function_cache_key_addition: int = 0,
) -> Tuple[gibbs_sampler.GibbsSamplerState, TensorLike, TensorLike]:
  """Structural Time Series (STS) methods for CausalImpact.

  Fits a Tensorflow Probability (TFP) structural time series model.

  Args:
    ci_data: CausalImpact data, used to fit a model.
    prior_level_sd: float: Prior standard deviation of the Gaussian random walk
      of the local level model. Expressed in terms of data standard deviations.
    seed: PRNG seed; see `tensorflow_probability.random.sanitize_seed` for
      details.
    num_results: See `fit_causalimpact`.
    num_warmup_steps: See `fit_causalimpact`.
    model: Optional custom model - `tfp.sts.StructuralTimeSeries`.
    dtype: The dtype to use throughout computation.
    seasons: See `fit_causalimpact`.
    experimental_tf_function_cache_key_addition: See `fit_causalimpact`.

  Returns:
    The model used, posterior samples, posterior means, posterior trajectories
  """
  if isinstance(seed, int):
    # Integers are stateful seeds, thus calling twice with the same seed will
    # return different results. A tuple version is stateless once passed in
    # sanitize_seed.
    seed = (0, seed)
  # While compiled samplers can take non-sanitized seeds, this
  # means our support for non-Tensor seeds will result in cache misses
  # when just the seed changes. Thus sanitize before tracing/compiling.
  seed = tfp.random.sanitize_seed(seed)

  design_matrix = None if ci_data.feature_ts is None else tf.convert_to_tensor(
      ci_data.feature_ts.values, dtype=dtype)

  # To combine posterior sampling with predictions, instead of just using
  # the pre-period, also use the post-period, but with all values being NaN.
  after_pre_period_length = ci_data.model_after_pre_data.shape[0]
  extended_outcome_ts = tfp.sts.MaskedTimeSeries(
      time_series=tf.concat([
          ci_data.outcome_ts.time_series,
          tf.fill(after_pre_period_length, tf.constant(
              float("nan"), dtype=dtype))
      ],
                            axis=0),
      is_missing=tf.concat([
          ci_data.outcome_ts.is_missing,
          tf.fill(after_pre_period_length, True)
      ],
                           axis=0))
  outcome_sd = tf.convert_to_tensor(
      np.nanstd(ci_data.outcome_ts.time_series, ddof=1), dtype=dtype)
  # TODO(colcarroll,jburnim): Move constants to be module-level and commented.
  r2 = 0.8
  if design_matrix is not None:
    observation_noise_scale = (
        tf.cast(tf.math.sqrt(1 - r2), dtype=dtype) * outcome_sd)
  else:
    observation_noise_scale = outcome_sd
  level_scale = tf.ones([], dtype=dtype) * prior_level_sd * outcome_sd
  seasonal_drift_scales = 0.01 * outcome_sd * tf.ones(
      shape=[len(seasons)], dtype=dtype)
  if ci_data.feature_ts is None:
    weights = tf.zeros([0], dtype=dtype)
  else:
    weights = tf.zeros(ci_data.feature_ts.shape[-1:], dtype=dtype)

  level = tf.zeros_like(extended_outcome_ts.time_series)
  slope = tf.zeros_like(extended_outcome_ts.time_series)

  samples, posterior_means, posterior_predictive_samples = _run_gibbs_sampler(
      sts_model=model,
      outcome_ts=extended_outcome_ts,
      outcome_sd=outcome_sd,
      design_matrix=design_matrix,
      num_results=num_results,
      num_warmup_steps=num_warmup_steps,
      observation_noise_scale=observation_noise_scale,
      level_scale=level_scale,
      seasonal_drift_scales=seasonal_drift_scales,
      weights=weights,
      level=level,
      slope=slope,
      seed=seed,
      dtype=dtype,
      seasons=seasons,
      # By passing in an extra Python integer, it will be part of the cache
      # key. Calling this twice with the same
      # experimental_tf_function_cache_key_addition will result in a cache hit,
      # while changing it to a (previously unused) different value will result
      # in a miss.
      experimental_tf_function_cache_key_addition=experimental_tf_function_cache_key_addition
  )
  return samples, posterior_means, posterior_predictive_samples


def _get_posterior_means_and_trajectories(sts_model, posterior_samples, seed):
  """Computes means and draws from posterior.

  Args:
    sts_model: Model that was used to fit the samples.
    posterior_samples: Draws from posterior of sts_model.
    seed: PRNG seed.

  Returns:
    Posterior means, posterior predictive samples
  """
  predictive_distributions = gibbs_sampler.one_step_predictive(
      sts_model,
      posterior_samples,
      # This gets us a sample per posterior sample, rather than just a fraction
      # of them.
      thin_every=1,
      use_zero_step_prediction=True)
  posterior_means = predictive_distributions.mean()
  # Returns a shape of [num_timesteps, num_posterior_samples].
  output_sample = predictive_distributions.components_distribution.sample(
      seed=seed)
  posterior_trajectories = tf.transpose(output_sample)
  return posterior_means, posterior_trajectories


def _compute_impact(
    posterior_means,
    posterior_trajectories,
    ci_data: cid.CausalImpactData,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Computes impact using posterior predictive distributions.

  Adds two attributes, both pd.DataFrames: one with timepoint-wise impact
  estimates, and another containing summaries over the post-treatment period.
  The former is used for plotting, the latter for summarizing overall impact.

  Args:
    posterior_means: Mean of posterior predictive trajectories.
    posterior_trajectories: Draws from the posterior predictive distribution.
      variables from sts_model.
    ci_data: an instance of a cid.CausalImpactData object.
    alpha: float, used to get confidence level for predictions.

  Returns:
    Two dataframes, the first with the impact timeseries, the second with a
    summary of that data.
  """
  # Check alpha.
  if not 0 < alpha < 1:
    raise ValueError("`alpha` must be between 0 and 1.")

  # Extract attributes of data object that we'll need for computing impact.
  observed_ts_pre = ci_data.pre_data[ci_data.outcome_column]
  observed_ts_post = ci_data.after_pre_data[ci_data.outcome_column]
  # Filter out data after the post-period.
  observed_ts_post = observed_ts_post.loc[
      (observed_ts_post.index >= ci_data.post_period[0])
      & (observed_ts_post.index <= ci_data.post_period[1])]
  observed_ts_full = pd.concat([observed_ts_pre, observed_ts_post], axis=0)

  # Get samples of posterior predictive trajectories and posterior means.
  # Ensure that lower quantile comes first.
  quantiles = (alpha / 2.0, 1.0 - (alpha / 2.0))
  posterior_trajectories, posterior_trajectory_summary = (
      _sample_posterior_predictive(
          posterior_means=posterior_means,
          posterior_trajectories=posterior_trajectories,
          ci_data=ci_data,
          quantiles=quantiles))

  # Use the posterior samples of trajectories to compute corresponding
  # trajectories for point, cumulative point, and relative effect estimates.
  trajectory_dict = _compute_impact_trajectories(
      posterior_trajectories,
      observed_ts_full,
      treatment_start=ci_data.post_period[0])

  # Create time series of mean and lower/upper quantiles for the point and
  # cumulative predictions.
  series = _compute_impact_estimates(
      posterior_trajectory_summary=posterior_trajectory_summary,
      trajectory_dict=trajectory_dict,
      observed_ts_full=observed_ts_full,
      ci_data=ci_data,
      quantiles=quantiles)

  # Create table to summarize results over the entire post-period.
  summary = _compute_summary(
      posterior_trajectory_summary=posterior_trajectory_summary,
      trajectory_dict=trajectory_dict,
      observed_ts_post=observed_ts_post,
      post_period=ci_data.post_period,
      quantiles=quantiles,
      alpha=alpha)
  return series, summary


def _sample_posterior_predictive(
    posterior_means: TensorLike,
    posterior_trajectories: TensorLike,
    ci_data: cid.CausalImpactData,
    quantiles: Tuple[float, float]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Samples from posterior predictive distribution and calculates summaries.

  Draws sample trajectories from the posterior predictive distribution and
  calculates the posterior mean for each time point. The posterior means are
  calculated separately for each component of the structural time series, so
  they exclude observation noise. Note that the mean of the posterior
  trajectories is, in expectation, equal to the posterior mean, but any given
  simulation with a finite number of MCMC iterations will include observation
  noise, so here we calculate the expectation without observation noise.
  This function also calculates posterior quantiles for each time point
  using the sampled trajectories at the values given in `quantiles`.

  Args:
    posterior_means: Means of the posterior predictive trajectories.
    posterior_trajectories: Draws from the posterior predictive distribution.
      variables from sts_model.
    ci_data: an instance of a cid.CausalImpactData object.
    quantiles: floats in (0, 1) giving the quantiles to calculate.

  Returns:
    Tuple of:
      posterior_trajectories: Sampled posterior trajectories. Columns represent
        the sampled posterior trajectories.
      posterior_trajectory_summary: Summary of the trajectories, including
        mean and specified quantiles. Columns represent the summary statistics.
    In both dataframes, the rows represent time and have the same index as the
    original data.
  """
  # Check that given quantiles are in (0, 1) and in order.
  if any((q < 0) | (q > 1) for q in quantiles):
    raise ValueError("All elements of `quantiles` must be in (0, 1). Got %s" %
                     quantiles)
  if quantiles[0] > quantiles[1]:
    raise ValueError("`quantiles` must be sorted in ascending order. Got %s" %
                     quantiles)

  # Compute posterior mean trajectories in the pre- and post-periods from the
  # predictive distributions.
  posterior_means = posterior_processing.process_posterior_quantities(
      ci_data, posterior_means, ["posterior_mean"])

  # Sample posterior trajectories from the posterior predictive distributions
  # and calculate the specified quantiles at each time point.
  posterior_trajectories = _package_posterior_trajectories(
      posterior_trajectories, ci_data)

  posterior_quantiles = posterior_processing.calculate_trajectory_quantiles(
      posterior_trajectories, "posterior", quantiles)

  # Combine posterior means and quantiles into a single dataframe by joining
  # on the index.
  posterior_trajectory_summary = posterior_means.join(posterior_quantiles)

  return (posterior_trajectories, posterior_trajectory_summary)


def _package_posterior_trajectories(
    posterior_trajectories: TensorLike,
    ci_data: cid.CausalImpactData) -> pd.DataFrame:
  """Repackages trajectories into a dataframe.

  NOTE: if the data in ci_data.model data was scaled, this function will undo
  the scaling so that the posterior trajectories are on the *original* scale.

  Args:
    posterior_trajectories: Sampled posterior predictive trajectories.
    ci_data: an instance of a cid.CausalImpactData object.

  Returns:
    pd.DataFrame with rows corresponding to time points and columns
      corresponding to each sampled posterior trajectory.
  """
  col_names = [
      f"sample_{i + 1}" for i in range(posterior_trajectories.shape[0])
  ]
  return posterior_processing.process_posterior_quantities(
      ci_data, posterior_trajectories, col_names)


def _compute_impact_trajectories(
    posterior_trajectories: pd.DataFrame, observed_ts_full: pd.Series,
    treatment_start: OutputDateType) -> Dict[str, pd.DataFrame]:
  """Computes trajectories of point and cumulative effects.

  Uses the sampled posterior predictive trajectories and observed outcome time
  series to compute corresponding sample trajectories of:
  point effects (observed - predicted),
  cumulative point effects (cumulative sum(observed - predicted))

  Args:
    posterior_trajectories: pd.DataFrame with rows corresponding to time points
      and columns corresponding to each sampled posterior trajectory.
    observed_ts_full: the observed outcome for the full time period.
    treatment_start: start time of the treatment.

  Returns:
    dictionary of the posterior prediction, point, and cumulative effect
    trajectories.
  """

  # Point effects are zero in the pre-period by definition. However, we do not
  # enforce this restriction here so that the pre-period point effect
  # estimates can be used as a model check (the uncertainty intervals should
  # include zero) and as a check of the treatment start date.
  # To calculate point effects, we first subtract the observed data from the
  # trajectories and then multiply by -1 to get the correct quantity. This
  # way, we avoid having to replicate the observed data to match the shape of
  # the sampled trajectories.
  point_effect_trajectories = posterior_trajectories.sub(
      observed_ts_full, axis=0).mul(-1)

  # Cumulative point effects are zero in the pre-period by definition.
  cum_effect_trajectories_base = point_effect_trajectories.copy()
  cum_effect_trajectories_base.loc[
      cum_effect_trajectories_base.index < treatment_start] = 0
  # Use axis=0 to calculate cumulative sum over time (down the rows) for each
  # sample (column).
  cum_effect_trajectories = cum_effect_trajectories_base.cumsum(axis=0)

  return {
      "predictions": posterior_trajectories,
      "point_effects": point_effect_trajectories,
      "cumulative_effects": cum_effect_trajectories
  }


def _compute_impact_estimates(posterior_trajectory_summary: pd.DataFrame,
                              trajectory_dict: Dict[str, pd.DataFrame],
                              observed_ts_full: pd.Series,
                              ci_data: cid.CausalImpactData,
                              quantiles: Tuple[float, float]) -> pd.DataFrame:
  """Computes timepoint-wise summaries of predictions and impact estimates.

  Takes the posterior means and trajectories calculated in predict() and
  computes point and cumulative point effects at each time point by
  subtracting from the observed data. This function calculates means and
  quantiles for point and cumulative point effects.

  The output of this function is the input to creating impact plots.

  Args:
    posterior_trajectory_summary: pd.DataFrame output of predict() containing
      the posterior means and quantiles of the predictions.
    trajectory_dict: dictionary of the posterior prediction trajectories and
      point, cumulative, and relative effect trajectories as calculated in
      _compute_impact_trajectories().
    observed_ts_full: the observed outcome for the full time period.
    ci_data: an instance of a cid.CausalImpactData object.
    quantiles: floats in (0, 1) giving the quantiles to calculate.

  Returns:
    pd.DataFrame with the same index as the original data and
      columns for mean, lower quantile, and upper quantile for the point
      predictions, point effects (i.e. observed data minus point prediction),
      and cumulative effects (i.e. cumulative observed data minus cumulative
      point prediction). It will also have columns for the start and end
      points of the pre and post periods.
  """
  # Using the posterior means and observed time series, calculate mean point
  # and cumulative point effects. Note that cumulative effects are zero in the
  # pre-period by definition.
  point_effects_mean = (
      observed_ts_full - posterior_trajectory_summary["posterior_mean"])
  point_effects_mean = point_effects_mean.to_frame(name="point_effects_mean")
  cum_effects_mean_base = point_effects_mean.copy()
  zero_inds = point_effects_mean.index < ci_data.post_period[0]
  cum_effects_mean_base.loc[zero_inds] = 0
  cum_effects_mean = cum_effects_mean_base.cumsum()
  cum_effects_mean.columns = ["cumulative_effects_mean"]

  # Calculate quantiles at each time point for point and cumulative effect
  # trajectories.
  point_effects_quantiles = posterior_processing.calculate_trajectory_quantiles(
      trajectory_dict["point_effects"], "point_effects", quantiles)
  cum_effects_quantiles = posterior_processing.calculate_trajectory_quantiles(
      trajectory_dict["cumulative_effects"], "cumulative_effects", quantiles)

  # Collect all dataframes
  impact_estimates = pd.concat([
      observed_ts_full.to_frame(name="observed"), posterior_trajectory_summary,
      point_effects_mean, point_effects_quantiles, cum_effects_mean,
      cum_effects_quantiles
  ],
                               axis=1)

  # The in-between period and after post-period should only have observed and
  # posteriors (to match original).
  impact_estimates.loc[
      ((impact_estimates.index > ci_data.pre_period[1]) &
       (impact_estimates.index < ci_data.post_period[0])) |
      (impact_estimates.index > ci_data.post_period[1]),
      impact_estimates.columns.difference(
          ["observed", "posterior_mean", "posterior_lower", "posterior_upper"]
      )] = np.nan

  # Where the observed was NaN, all other values should be NaN (and not zero).
  # This follows from the fact there is no real value to compare to.
  impact_estimates.loc[
      np.isnan(impact_estimates["observed"]),
      impact_estimates.columns.difference(
          ["observed", "posterior_mean", "posterior_lower", "posterior_upper"]
      )] = np.nan

  # The impact estimates so far of the range [pre_period[0], end_of_input_data],
  # but we actually want to return the original values before the pre-period.
  # Thus change the index to match the original data, then copy over the entire
  # original timeseries.
  impact_estimates = impact_estimates.reindex(
      ci_data.data.index, copy=False, fill_value=np.nan)
  impact_estimates["observed"] = ci_data.data[ci_data.outcome_column]

  # Add the pre/post period dates as columns for easier plotting.
  impact_estimates["pre_period_start"] = ci_data.pre_period[0]
  impact_estimates["pre_period_end"] = ci_data.pre_period[1]
  impact_estimates["post_period_start"] = ci_data.post_period[0]
  impact_estimates["post_period_end"] = ci_data.post_period[1]

  return impact_estimates


def _compute_summary(posterior_trajectory_summary: pd.DataFrame,
                     trajectory_dict: Dict[str, pd.DataFrame],
                     observed_ts_post: pd.Series, post_period: OutputPeriodType,
                     quantiles: Tuple[float,
                                      float], alpha: float) -> pd.DataFrame:
  """Computes summary statistics of the average and cumulative impact.

  Calculates average and cumulative summaries of counterfactual forecasts and
  pointwise impact estimates over the post-treatment period. Note that because
  the average/sum of a quantile isn't the same as the quantile of an
  average/sum, we can't just use the output of _compute_impact_estimates()
  and instead need to use the sampled trajectories to get the correct values.

  Args:
    posterior_trajectory_summary: pd.DataFrame output of predict() containing
      the posterior means and quantiles of the predictions.
    trajectory_dict: dictionary of the posterior prediction trajectories and
      point and cumulative effect trajectories as calculated in
      _compute_impact_trajectories().
    observed_ts_post: the observed outcome for the post-period.
    post_period: Range of the post-period.
    quantiles: floats in (0, 1) giving the quantiles to calculate.
    alpha: float specifying the quantiles.

  Returns:
    pd.DataFrame with the same index as the original data and
      columns for mean, lower quantile, and upper quantile for the point
      predictions, point effects (i.e. observed data minus point prediction),
      and cumulative effects (i.e. cumulative observed data minus cumulative
      point prediction).
  """

  # Restrict the posterior mean trajectory and sampled impact trajectories to
  # the post-period.
  posterior_mean = posterior_trajectory_summary.loc[
      (posterior_trajectory_summary.index >= post_period[0]) &
      (posterior_trajectory_summary.index <= post_period[1]), "posterior_mean"]
  trajectory_dict = {
      k: v.loc[(v.index >= post_period[0]) & (v.index <= post_period[1])]
      for k, v in trajectory_dict.items()
  }

  # PREDICTIONS
  # Calculate average and cumulative values of the posterior mean -- i.e. the
  # counterfactual predictions -- over the post-period.
  average_prediction = posterior_mean.mean()
  cumulative_prediction = posterior_mean.sum()

  # Use the sampled posterior trajectories to get uncertainty intervals for
  # the above quantities. We'll calculate the mean (sum) of the trajectories
  # *across the post-period*. Since we want to summarize over time for each
  # trajectory, we use axis=0.
  pred_trajectories_mean = trajectory_dict["predictions"].mean(axis=0)
  pred_trajectories_sum = trajectory_dict["predictions"].sum(axis=0)
  avg_pred_lower, avg_pred_upper = pred_trajectories_mean.quantile(quantiles)
  cum_pred_lower, cum_pred_upper = pred_trajectories_sum.quantile(quantiles)

  # POINT EFFECTS
  # Calculate the average and cumulative point effects. Use the corresponding
  # sampled impact trajectories to calculate lower/upper uncertainty
  # intervals.
  average_point_effect = observed_ts_post.mean() - average_prediction
  pt_eff_trajectories_mean = trajectory_dict["point_effects"].mean(axis=0)
  avg_pt_eff_lower, avg_pt_eff_upper = (
      pt_eff_trajectories_mean.quantile(quantiles))

  cumulative_point_effect = observed_ts_post.sum() - cumulative_prediction
  pt_eff_trajectories_sum = trajectory_dict["point_effects"].sum(axis=0)
  cum_pt_eff_lower, cum_pt_eff_upper = (
      pt_eff_trajectories_sum.quantile(quantiles))

  # RELATIVE EFFECTS
  # Calculate the average and cumulative relative effects. Use the
  # corresponding sampled impact trajectories to calculate lower/upper
  # uncertainty intervals. Note that the average and cumulative relative
  # effects are mathematically identical.
  rel_eff_trajectories_mean = (
      observed_ts_post.sum() / pred_trajectories_sum - 1.)
  avg_rel_eff_lower, avg_rel_eff_upper = (
      rel_eff_trajectories_mean.quantile(quantiles))

  cumulative_relative_effect = rel_eff_trajectories_mean
  cum_rel_eff_lower, cum_rel_eff_upper = (
      cumulative_relative_effect.quantile(quantiles))

  # Calculate summary statistics of the average and cumulative impact over the
  # post-period.
  summary_dict = {
      "actual": {
          "average": observed_ts_post.mean(),
          "cumulative": observed_ts_post.sum()
      },
      "predicted": {
          "average": average_prediction,
          "cumulative": cumulative_prediction
      },
      "predicted_lower": {
          "average": avg_pred_lower,
          "cumulative": cum_pred_lower
      },
      "predicted_upper": {
          "average": avg_pred_upper,
          "cumulative": cum_pred_upper
      },
      "predicted_sd": {
          "average": pred_trajectories_mean.std(),
          "cumulative": pred_trajectories_sum.std()
      },
      "abs_effect": {
          "average": average_point_effect,
          "cumulative": cumulative_point_effect
      },
      "abs_effect_lower": {
          "average": avg_pt_eff_lower,
          "cumulative": cum_pt_eff_lower
      },
      "abs_effect_upper": {
          "average": avg_pt_eff_upper,
          "cumulative": cum_pt_eff_upper
      },
      "abs_effect_sd": {
          "average": pt_eff_trajectories_mean.std(),
          "cumulative": pt_eff_trajectories_sum.std()
      },
      "rel_effect": {
          "average": np.mean(cumulative_relative_effect),
          "cumulative": np.mean(cumulative_relative_effect)
      },
      "rel_effect_lower": {
          "average": avg_rel_eff_lower,
          "cumulative": cum_rel_eff_lower
      },
      "rel_effect_upper": {
          "average": avg_rel_eff_upper,
          "cumulative": cum_rel_eff_upper
      },
      "rel_effect_sd": {
          "average": cumulative_relative_effect.std(),
          "cumulative": cumulative_relative_effect.std()
      }
  }
  summary_df = pd.DataFrame(summary_dict)

  # Add a column for the one-sided "p-value" based on the cumulative effect.
  # Note that the p-value will be the same for both the "average" and
  # "cumulative" rows. We concatenate the observed cumulative outcome with the
  # sampled ones to ensure that the p-value will fall within (0, 1).
  observed_cumulative_outcome = observed_ts_post.sum()
  sampled_cumulative_outcomes = pd.concat(
      [pred_trajectories_sum,
       pd.Series(observed_cumulative_outcome)], axis=0)
  prop_obs_lessthan = (observed_cumulative_outcome <=
                       sampled_cumulative_outcomes).mean()
  prop_obs_greaterthan = (observed_cumulative_outcome >=
                          sampled_cumulative_outcomes).mean()
  p_value = min(prop_obs_lessthan, prop_obs_greaterthan)
  summary_df["p_value"] = p_value
  summary_df["alpha"] = alpha

  return summary_df
