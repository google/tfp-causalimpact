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

"""Tests for causalimpact_lib.py."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import causalimpact as ci
from causalimpact import causalimpact_lib
from causalimpact import data as cid
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
import tensorflow as tf


CURR_PATH = os.path.dirname(__file__)
TEST_DATA = os.path.join(CURR_PATH, "testdata", "data.csv")


def _create_test_data(treat_amt, treat_index, num_timesteps=100):
  ar = np.r_[1, 0.9]
  ma = np.array([1])
  arma_process = ArmaProcess(ar, ma)
  x = 100 + arma_process.generate_sample(nsample=num_timesteps)
  y = 1.2 * x + np.random.normal(size=(num_timesteps))
  test_data = pd.DataFrame({"y": y, "x": x}, columns=["y", "x"])
  date_index = pd.date_range("2018-01-01", periods=len(y), freq="D")
  test_data.index = date_index
  test_data.loc[test_data.index > date_index[treat_index], "y"] += treat_amt
  return test_data


def _create_impact_data():
  # Make some fake impact data.
  n_time_points = 100
  n_samples = 10
  treat_start_index = 50
  post_period_length = n_time_points - treat_start_index
  treatment_effect = 5.
  cumulative_treatment_effect = treatment_effect * post_period_length
  index = pd.date_range("2018-01-01", periods=n_time_points, freq="D")
  treatment_start = index[treat_start_index]
  pre_period = (index[0], index[treat_start_index - 1])
  post_period = (index[treat_start_index], index[-1])
  colnames = ["sample_" + str(i) for i in range(n_samples)]

  # Make fake data that is just the sequence 0:(n_time_points-1), with an
  # additive treatment effect of 5.
  observed_ts_full = pd.Series(
      np.arange(n_time_points).astype(float), index=index)
  observed_ts_full[treat_start_index:] += treatment_effect

  # The fake "sampled" posterior trajectories are just the sequence
  # 0:(n_time_points-1), i.e. the observed data minus the treatment effect of
  # 5. Note that the posterior trajectories are identical to each other.
  posterior_trajectories = pd.DataFrame(
      pd.concat([pd.Series(np.arange(n_time_points))] * n_samples, axis=1))
  posterior_trajectories.columns = colnames
  posterior_trajectories.index = index

  # The point effect trajectories are just zero in the pre-period and the
  # treatment effect in the post-period.
  one_point_trajectory = pd.Series(np.repeat([0, treatment_effect], 50))
  one_point_trajectory.index = index
  point_trajectories = pd.DataFrame(
      pd.concat([one_point_trajectory] * n_samples, axis=1))
  point_trajectories.columns = colnames

  # Cumulative effect trajectories are the cumulative sum of the expected point
  # trajectories over the post-period (and zero in the pre-period).
  cumulative_trajectories = point_trajectories.cumsum(axis=0)

  # Combine all trajectories.
  trajectory_dict = {
      "predictions": posterior_trajectories,
      "point_effects": point_trajectories,
      "cumulative_effects": cumulative_trajectories
  }

  # Since the fake trajectories are identical, the quantiles, means, and
  # will be equal to a single trajectory. Here we
  # create a dataframe with the summary statistics for the posterior, point, and
  # cumulative trajectories.
  sample_cols = ["sample_1", "sample_2", "sample_3"]
  summary_stats = ["mean", "lower", "upper"]
  posterior_trajectory_summary = posterior_trajectories[sample_cols].copy()
  posterior_trajectory_summary.columns = [
      "posterior_" + stat for stat in summary_stats
  ]
  point_trajectory_summary = point_trajectories[sample_cols].copy()
  point_trajectory_summary.columns = [
      "point_effects_" + stat for stat in summary_stats
  ]
  cumulative_trajectory_summary = cumulative_trajectories[sample_cols].copy()
  cumulative_trajectory_summary.columns = [
      "cumulative_effects_" + stat for stat in summary_stats
  ]

  # Join all trajectory summaries on the time index.
  trajectory_summary = pd.concat([
      observed_ts_full, posterior_trajectory_summary, point_trajectory_summary,
      cumulative_trajectory_summary
  ],
                                 axis=1)
  trajectory_summary.rename(columns={0: "observed"}, inplace=True)

  # For the impact summary table, we need to compute the expected post-period
  # predictions, which is just the sequence treat_start_index:(n_time_points-1),
  # i.e. the observed data minus the treatment effect.
  expected_pred = np.arange(start=treat_start_index, stop=n_time_points)

  # Compute impact summary. Because each "sampled" trajectory for this fake data
  # is identical, the lower/upper values for each quantity are identical to
  # their corresponding point estimates and all standard deviations are zero.
  expected_summary_dict = {
      "actual": {
          "average": observed_ts_full[treat_start_index:].mean(),
          "cumulative": observed_ts_full[treat_start_index:].sum()
      },
      "predicted": {
          "average": expected_pred.mean(),
          "cumulative": expected_pred.sum()
      },
      "predicted_lower": {
          "average": expected_pred.mean(),
          "cumulative": expected_pred.sum()
      },
      "predicted_upper": {
          "average": expected_pred.mean(),
          "cumulative": expected_pred.sum()
      },
      "predicted_sd": {
          "average": 0.,
          "cumulative": 0.
      },
      "abs_effect": {
          "average": treatment_effect,
          "cumulative": cumulative_treatment_effect
      },
      "abs_effect_lower": {
          "average": treatment_effect,
          "cumulative": cumulative_treatment_effect
      },
      "abs_effect_upper": {
          "average": treatment_effect,
          "cumulative": cumulative_treatment_effect
      },
      "abs_effect_sd": {
          "average": 0.,
          "cumulative": 0.
      },
      "rel_effect": {
          "average": treatment_effect / expected_pred.mean(),
          "cumulative": (cumulative_treatment_effect) / expected_pred.sum()
      },
      "rel_effect_lower": {
          "average": treatment_effect / expected_pred.mean(),
          "cumulative": (cumulative_treatment_effect) / expected_pred.sum()
      },
      "rel_effect_upper": {
          "average": treatment_effect / expected_pred.mean(),
          "cumulative": (cumulative_treatment_effect) / expected_pred.sum()
      },
      "rel_effect_sd": {
          "average": 0.,
          "cumulative": 0.
      }
  }
  expected_summary = pd.DataFrame(expected_summary_dict)

  expected_summary["p_value"] = 1. / (n_samples + 1)
  expected_summary["alpha"] = 0.05

  return {
      "treatment_start": treatment_start,
      "pre_period": pre_period,
      "post_period": post_period,
      "observed_ts_full": observed_ts_full,
      "observed_ts_post": observed_ts_full[treat_start_index:],
      "posterior_summary": posterior_trajectory_summary,
      "trajectory_dict": trajectory_dict,
      "trajectory_summary": trajectory_summary,
      "expected_summary": expected_summary
  }


class CausalImpactTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(CausalImpactTest, cls).setUpClass()

    # Read in sample data and set column "t" as the index
    with open(TEST_DATA, "r") as fl:
      df = pd.read_csv(fl)
    df = df.set_index(pd.to_datetime(df["t"]))
    df.drop(columns=["t"], inplace=True)
    # Explicitly set the frequency to match the input data, which is required.
    df.index.freq = "10s"
    df.loc[df.index[[1, 3, 7]], "y"] = np.nan
    cls.data = df
    treatment_start_index = 60
    cls.pre_period = (df.index[0], df.index[treatment_start_index - 1])
    cls.post_period = (df.index[treatment_start_index], df.index[-1])
    cls.treatment_start = cls.post_period[0]

  def testShortestPeriodAfterPrePeriod(self):
    ci_analysis = ci.fit_causalimpact(
        data=self.data,
        pre_period=(self.data.index[0], self.data.index[-2]),
        post_period=(self.data.index[-1], self.data.index[-1]),
        inference_options=ci.InferenceOptions(num_results=10),
        seed=(1, 2))
    self.assertIsNotNone(ci_analysis)

  def testUnexpectedKwargsRaisesAnError(self):
    """kwargs are accepted for experimental args, but typos should be caught."""
    with self.assertRaises(TypeError):
      ci.fit_causalimpact(
          some_unknown_arg=3,
          data=self.data,
          pre_period=(self.data.index[0], self.data.index[-2]),
          post_period=(self.data.index[-1], self.data.index[-1]),
          inference_options=ci.InferenceOptions(num_results=10),
          seed=(1, 2))

  @parameterized.named_parameters(
      {
          "testcase_name": "0.01",
          "prior_level_sd": 0.01,
      }, {
          "testcase_name": "0.1",
          "prior_level_sd": 0.1,
      }, {
          "testcase_name": "0.5",
          "prior_level_sd": 0.5,
      })
  def testPriorLevelSdIsUsed(self, prior_level_sd):
    seed = (0, 0)
    treatment_start = 20
    data = self.data
    ci_analysis = ci.fit_causalimpact(
        data=data,
        pre_period=(data.index[0], data.index[treatment_start - 1]),
        post_period=(data.index[treatment_start], data.index[-1]),
        inference_options=ci.InferenceOptions(
            num_results=100, num_warmup_steps=100),
        model_options=ci.ModelOptions(prior_level_sd=prior_level_sd),
        seed=seed)
    # It is a little surprising this passes -- it may be because we are Learning
    # on a small amount of data, but it also may be because the InverseGamma
    # prior is too strong.
    np.testing.assert_allclose(
        np.mean(ci_analysis.posterior_samples.level_scale),
        prior_level_sd,
        atol=0.2 * prior_level_sd)

  def testModelTrainingNoDatetimeIndexSucceeds(self):
    seed = (0, 0)
    data = self.data.copy()
    data.index = np.arange(data.shape[0])
    treatment_start = 20
    ci_analysis = ci.fit_causalimpact(
        data=data,
        pre_period=(data.index[0], data.index[treatment_start - 1]),
        post_period=(data.index[treatment_start], data.index[-1]),
        inference_options=ci.InferenceOptions(num_results=10),
        seed=seed)
    self.assertIsNotNone(ci_analysis)

  def testInterceptIsIncluded(self):
    seed = (0, 0)
    ci_analysis = ci.fit_causalimpact(
        data=self.data,
        pre_period=self.pre_period,
        post_period=self.post_period,
        inference_options=ci.InferenceOptions(num_results=10),
        seed=seed)
    # TEST_DATA.txt has 2 features, plus one intercept
    self.assertEqual(ci_analysis.posterior_samples.weights.shape[-1], 3)

  def testModelTrainingNoCovariates(self):
    ci_data = cid.CausalImpactData(
        self.data["y"],
        pre_period=self.pre_period,
        post_period=self.post_period,
        dtype=tf.float64)
    sts_model = causalimpact_lib._build_default_gibbs_model(
        ci_data.feature_ts,
        ci_data.outcome_ts,
        outcome_sd=tf.constant(1., dtype=tf.float64),
        level_scale=tf.constant(0.01, dtype=tf.float64),
        dtype=tf.float64,
        seasons=[])
    model_params = [p.name for p in sts_model.parameters]
    # TODO(colcarroll,jburnim): Consider how to make these tests more robust,
    # as they will currently break on name changes in the underlying libraries.
    self.assertIn("observation_noise_scale", model_params)
    self.assertTrue(
        any(["LocalLevel/_level_scale" in param for param in model_params]) or
        # Name used by GibbsSampler.
        any(["local_level/_level_scale" in param for param in model_params]))

  def testModelTrainingWithCovariates(self):
    seed = (1, 1)
    ci_data = cid.CausalImpactData(
        self.data,
        pre_period=self.pre_period,
        post_period=self.post_period,
        dtype=tf.float32)
    posterior_samples, *_ = causalimpact_lib._train_causalimpact_sts(
        ci_data=ci_data,
        seed=seed,
        num_warmup_steps=100,
        num_results=10,
        prior_level_sd=0.01,
        dtype=tf.float32,
        seasons=[])

    self.assertFalse(np.any(np.isnan(posterior_samples.level)))
    self.assertTrue(np.all(posterior_samples.observation_noise_scale <= 1.2))
    self.assertTrue(np.all(posterior_samples.level_scale <= 1))
    self.assertFalse(np.any(np.isnan(posterior_samples.weights)))

  def testPredictionDims_NoCovars(self):
    num_results = 10
    impact = ci.fit_causalimpact(
        self.data["y"],
        inference_options=ci.InferenceOptions(num_results=num_results),
        pre_period=self.pre_period,
        post_period=self.post_period,
    )
    self.assertIsNotNone(impact.posterior_samples.observation_noise_scale)
    self.assertIsNotNone(impact.posterior_samples.level_scale)
    self.assertIsNotNone(impact.posterior_samples.level)
    # There are no covariates, expect this to not be set.
    self.assertIsNone(impact.posterior_samples.weights)

    # Check that time indices are correct.
    self.assertTrue(impact.series.index.equals(self.data.index))

    # Check that we got the right number of samples.
    self.assertEqual(impact.posterior_samples.observation_noise_scale.shape[0],
                     num_results)

  def testPredictionDims_WithCovars(self):
    num_results = 10
    impact = ci.fit_causalimpact(
        self.data,
        inference_options=ci.InferenceOptions(num_results=num_results),
        pre_period=self.pre_period,
        post_period=self.post_period,
    )

    # Check that time indices are correct.
    self.assertTrue(impact.series.index.equals(self.data.index))

    # Check that we got the right number of samples.
    self.assertEqual(impact.posterior_samples.observation_noise_scale.shape[0],
                     num_results)
    # Confirm that for a low dimensional example, the weights are never 0,
    # since the default prior sets the nonzero probability to 1 for
    # fewer than 3 dimensions (in which case overfitting is not a great fear).
    self.assertEqual((impact.posterior_samples.weights.numpy() == 0).sum(), 0)

  def testComputeImpactTrajectories(self):
    """Test for _compute_impact_trajectories()."""
    # Load test data.
    test_impact_data = _create_impact_data()
    expected_trajectories = test_impact_data["trajectory_dict"]

    # Pull out the posterior trajectories. They are an input to
    # _compute_impact_trajectories() and returned unmodified in its output
    # (just renamed to "predictions").
    posterior_trajectories = expected_trajectories["predictions"]

    trajectory_dict = causalimpact_lib._compute_impact_trajectories(
        posterior_trajectories=posterior_trajectories,
        observed_ts_full=test_impact_data["observed_ts_full"],
        treatment_start=test_impact_data["treatment_start"])

    expected_keys = ["predictions", "point_effects", "cumulative_effects"]
    for k in expected_keys:
      self.assertTrue(trajectory_dict[k].equals(expected_trajectories[k]))

  def testComputeImpactEstimates(self):
    """Test for _compute_impact_estimates()."""
    # Load test data.
    test_impact_data = _create_impact_data()
    expected_trajectories = test_impact_data["trajectory_dict"]
    expected_impact_estimates = test_impact_data["trajectory_summary"]

    # Compute trajectories of point, cumulative, and relative effects using
    # test data.
    trajectory_dict = causalimpact_lib._compute_impact_trajectories(
        posterior_trajectories=expected_trajectories["predictions"],
        observed_ts_full=test_impact_data["observed_ts_full"],
        treatment_start=test_impact_data["treatment_start"])
    impact_estimates = causalimpact_lib._compute_impact_estimates(
        posterior_trajectory_summary=test_impact_data["posterior_summary"],
        trajectory_dict=trajectory_dict,
        observed_ts_full=test_impact_data["observed_ts_full"],
        ci_data=cid.CausalImpactData(
            data=test_impact_data["observed_ts_full"],
            pre_period=test_impact_data["pre_period"],
            post_period=test_impact_data["post_period"],
        ),
        quantiles=(0.025, 0.975))

    # Make sure the impact estimates are what we expect. Use column names in
    # expected_impact_estimates to make sure the columns are in the same order
    # between the two dataframes.
    col_names = expected_impact_estimates.columns
    pd.testing.assert_frame_equal(impact_estimates[col_names],
                                  expected_impact_estimates)

  def testComputeImpactSummary(self):
    """Test for _compute_summary()."""

    # Load test data.
    test_impact_data = _create_impact_data()
    expected_trajectories = test_impact_data["trajectory_dict"]
    posterior_trajectory_summary = (test_impact_data["trajectory_summary"])
    expected_summary = test_impact_data["expected_summary"]

    trajectory_dict = causalimpact_lib._compute_impact_trajectories(
        posterior_trajectories=expected_trajectories["predictions"],
        observed_ts_full=test_impact_data["observed_ts_full"],
        treatment_start=test_impact_data["treatment_start"])

    # Calculate impact summary using test data.
    summary = causalimpact_lib._compute_summary(
        posterior_trajectory_summary=posterior_trajectory_summary,
        trajectory_dict=trajectory_dict,
        observed_ts_post=test_impact_data["observed_ts_post"],
        post_period=test_impact_data["post_period"],
        quantiles=(0.025, 0.975),
        alpha=0.05)

    # Make sure the columns and indices are what we expect.
    self.assertSetEqual(set(summary.columns), set(expected_summary.columns))
    self.assertTrue(summary.index.equals(pd.Index(["average", "cumulative"])))

    # Make sure the values are what we expect.
    pd.testing.assert_frame_equal(expected_summary, summary)

  @parameterized.named_parameters(
      {
          "testcase_name": "TupleInts",
          "seed": (13, 37),
      }, {
          "testcase_name": "Int",
          "seed": 14
      })
  def testEvaluate(self, seed):
    """Test for evaluate().

    The workhorse functions of evaluate() are tested in the two previous tests,
    testComputeImpactEstimates() and testComputeImpactSummary(). Here we
    simply test that evaluate() creates the expected attributes (series and
    summary).

    Args:
     seed: Seed to train with.
    """
    treat_index = 50
    treat_amt = 5
    test_data = _create_test_data(treat_amt, treat_index)
    impact = ci.fit_causalimpact(
        test_data.copy(),
        pre_period=(test_data.index[0], test_data.index[treat_index - 1]),
        post_period=(test_data.index[treat_index], test_data.index[-1]),
        inference_options=ci.InferenceOptions(num_results=10),
        seed=seed)
    self.assertIsInstance(impact.series, pd.DataFrame)
    self.assertIsInstance(impact.summary, pd.DataFrame)

    # Check that `evaluate` is deterministic (when using a stateless seed).
    new_impact = ci.fit_causalimpact(
        test_data,
        pre_period=(test_data.index[0], test_data.index[treat_index - 1]),
        post_period=(test_data.index[treat_index], test_data.index[-1]),
        inference_options=ci.InferenceOptions(num_results=10),
        seed=seed)

    pd.testing.assert_frame_equal(impact.series, new_impact.series)
    pd.testing.assert_frame_equal(impact.summary, new_impact.summary)

  @parameterized.named_parameters(
      {
          "testcase_name": "NoTimeAfterPostPeriod",
          "time_after_post_period": False,
      }, {
          "testcase_name": "TimeAfterPostPeriod",
          "time_after_post_period": True
      })
  def testSummary(self, time_after_post_period):
    """Verifies that time after the post period does not change the summary."""
    treat_index = 50
    treat_amt = 5
    timesteps_before_post_period_ends = 100
    # In the extra time case, just add an extra set of points but do NOT
    # move the post-period.
    num_timesteps = timesteps_before_post_period_ends + (
        50 if time_after_post_period else 0)
    test_data = _create_test_data(
        treat_amt, treat_index, num_timesteps=num_timesteps)
    pre_period = (test_data.index[0], test_data.index[treat_index - 1])
    post_period = (test_data.index[treat_index],
                   test_data.index[timesteps_before_post_period_ends - 1])
    impact = ci.fit_causalimpact(
        test_data.copy(),
        pre_period=pre_period,
        post_period=post_period,
        inference_options=ci.InferenceOptions(num_results=10),
        seed=0)
    # This confirms that the total cumulative effect is approximately correct.
    # If we did not respect the end of post periods, it would be double.
    np.testing.assert_allclose(
        impact.summary.loc["cumulative", "abs_effect"], 250, rtol=0.2)

  def testNonAlignedStartTime(self):
    """Verifies that pre/post periods do not need to be aligned with the data."""
    n_time_steps = 100
    treat_start = 50
    true_effect = 5.
    y = np.random.normal(size=n_time_steps, scale=0.0001)
    y[treat_start:] += true_effect
    date_index = pd.date_range("2018-01-07", periods=n_time_steps, freq="W")
    test_data = pd.DataFrame({"y": y}, index=date_index)

    analysis = ci.fit_causalimpact(
        test_data,
        # Choose dates not aligned to 2018-01-07 on a weekly cadence.
        pre_period=("2018-01-10", "2018-01-30"),
        post_period=("2018-02-02", "2018-02-23"),
        inference_options=ci.InferenceOptions(num_results=10))
    # Verify that we have no information set before post period starts,
    # rounding up.
    self.assertEqual(
        # TODO(colcarroll,jburnim): This should be NaN, not 0.
        0,
        analysis.series.loc[pd.to_datetime("2018-01-28"),
                            "cumulative_effects_mean"])
    self.assertNotEqual(
        0, analysis.series.loc[pd.to_datetime("2018-02-04"),
                               "cumulative_effects_mean"])

  def testGapBetweenPreAndPostPeriod(self):
    """Verifies that pre/post periods do not need to be aligned with the data."""
    n_time_steps = 20
    treat_start = 50
    true_effect = 5.
    y = np.random.normal(size=n_time_steps, scale=0.0001)
    y[treat_start:] += true_effect
    date_index = pd.date_range("2022-01-07", periods=n_time_steps, freq="D")
    data_period = (date_index[0], date_index[-1])
    test_data = pd.DataFrame({"y": y}, index=date_index)

    before_pre_period = (data_period[0], "2022-01-08")
    pre_period = ("2022-01-09", "2022-01-12")
    inbetween_period = ("2022-01-13", "2022-01-14")
    # Post period finishes before all timesteps provided, to test that
    # behaviour.
    post_period = ("2022-01-15", "2022-01-20")
    after_post_period = ("2022-01-21", data_period[1])

    analysis = ci.fit_causalimpact(
        test_data,
        pre_period=pre_period,
        post_period=post_period,
        inference_options=ci.InferenceOptions(num_results=10))
    series = analysis.series
    # Drop the non-number values for easier testing.
    series.drop([
        "pre_period_start", "pre_period_end", "post_period_start",
        "post_period_end"
    ],
                inplace=True,
                axis="columns")
    with self.subTest("IndexMathes"):
      # Expect data for all points, even outside the pre- and post-period range.
      pd.testing.assert_index_equal(series.index, date_index)

    with self.subTest("EntirePeriod"):
      # Verify observed is returned, even outside of the pre- and post-period
      # range.
      self.assertTrue(series.loc[data_period[0]:data_period[1],
                                 ["observed"]].notna().all(axis=None))

    with self.subTest("BeforePreperiod"):
      # Only observed should be set before pre-period.
      self.assertTrue(
          series.loc[before_pre_period[0]:before_pre_period[1],
                     series.columns.difference(["observed"])].isna().all(
                         axis=None))

    with self.subTest("Pre-period"):
      # Verify posterior and point effects have values.
      self.assertTrue(series.loc[pre_period[0]:pre_period[1], [
          "posterior_mean", "posterior_lower", "posterior_upper",
          "point_effects_mean", "point_effects_lower", "point_effects_upper"
      ]].notna().all(axis=None))
      # Verify cumulative values are zero.
      self.assertTrue(series.loc[pre_period[0]:pre_period[1], [
          "cumulative_effects_mean", "cumulative_effects_lower",
          "cumulative_effects_upper"
      ]].eq(0).all(axis=None))

    with self.subTest("Inbetween-period"):
      # Verify posterior values are returned.
      self.assertTrue(series.loc[
          inbetween_period[0]:inbetween_period[1],
          ["posterior_mean", "posterior_lower", "posterior_upper"]].notna().all(
              axis=None))
      # Verify no point and cumulative values.
      self.assertTrue(series.loc[inbetween_period[0]:inbetween_period[1], [
          "point_effects_mean", "point_effects_lower", "point_effects_upper",
          "cumulative_effects_mean", "cumulative_effects_lower",
          "cumulative_effects_upper"
      ]].isna().all(axis=None))

    with self.subTest("Post-period"):
      # Verify that all values are returned.
      self.assertTrue(
          series.loc[post_period[0]:post_period[1]].notna().all(axis=None))

    with self.subTest("AfterPostperiod"):
      # Only the observed and posterior should be set after post-period.
      expected_columns = [
          "observed", "posterior_mean", "posterior_lower", "posterior_upper"
      ]
      self.assertTrue(
          series.loc[after_post_period[0]:after_post_period[1],
                     series.columns.difference(expected_columns)].isna().all(
                         axis=None))
      self.assertTrue(series.loc[after_post_period[0]:after_post_period[1],
                                 expected_columns].notna().all(axis=None))

  @parameterized.named_parameters(
      {
          "testcase_name": "float32",
          "dtype": tf.float32
      }, {
          "testcase_name": "float64",
          "dtype": tf.float64
      })
  def testNumericImpactValues(self,
                              dtype=tf.float32):
    """Test for numeric values of evaluate().

    This test uses simulated data with a large treatment effect and a large
    number of MCMC draws to check that the numeric values of the estimated
    effects and their uncertainty intervals are close to what we'd expect.

    Args:
      dtype: The dtype to use throughout computation.
    """
    n_time_steps = 100
    treat_start = 50
    true_effect = 5.
    y = np.random.normal(size=n_time_steps, scale=0.0001)
    y[treat_start:] += true_effect
    # TODO(colcarroll): Add a version of this test that does not standardize
    # data and by adding (say) 100 to `y`, then asserting that
    # `mod.series[0, posterior_lower] < y[0]`, and that
    # y[0] < mod.series[0, posterior_upper]`
    date_index = pd.date_range("2018-01-01", periods=n_time_steps, freq="D")
    test_data = pd.DataFrame({"y": y}, index=date_index)
    impact = ci.fit_causalimpact(
        test_data,
        pre_period=(test_data.index[0], test_data.index[treat_start - 1]),
        post_period=(test_data.index[treat_start], test_data.index[-1]),
        inference_options=ci.InferenceOptions(num_results=1000),
        data_options=ci.DataOptions(dtype=dtype))
    summary = impact.summary
    # The true absolute effects, as average and sum over post period.
    true_abs_effects = (true_effect, true_effect * (n_time_steps - treat_start))
    np.testing.assert_allclose(
        summary["abs_effect"], true_abs_effects, rtol=0.001, atol=0.001)

    # See that interval widths (relative to absolute effects) are about 0.
    relative_interval_widths = (
        (summary["abs_effect_upper"] - summary["abs_effect_lower"]) /
        summary["abs_effect"])
    self.assertLessEqual(relative_interval_widths["average"], 0.01)
    self.assertLessEqual(relative_interval_widths["cumulative"], 0.01)

  def testNumericImpactValuesWithSeasonality(self):
    """Verifies that seasonality assists with inference.

    This is not intended to do a detailed analysis of the inference, just that
    there is difference when seasonal effects are modeled when the data
    does have a true underlying effect.
    """
    dtype = tf.float32
    n_time_steps = 300
    treat_start = 290
    true_effect = 2.5
    every_five_effect = [
        [8., 8., 4., 3., -4.][x % 5] for x in range(n_time_steps)
    ]
    every_seven_effect = [
        10 * [1., 4., 5., 2., -1., -2., -3.][x % 7] for x in range(n_time_steps)
    ]
    every_eight_effect = [
        [1., 1., 3., 3., 4.5, 2.0, -7., 0.][x % 8] for x in range(n_time_steps)
    ]
    y = (
        np.random.normal(size=n_time_steps, scale=0.4) + every_seven_effect +
        every_five_effect + every_eight_effect)

    y[treat_start:] += true_effect
    date_index = pd.date_range("2018-01-01", periods=n_time_steps, freq="D")
    test_data = pd.DataFrame({"y": y}, index=date_index)
    impact_without_season = ci.fit_causalimpact(
        test_data,
        pre_period=(test_data.index[0], test_data.index[treat_start - 1]),
        post_period=(test_data.index[treat_start], test_data.index[-1]),
        inference_options=ci.InferenceOptions(num_results=1000),
        data_options=ci.DataOptions(dtype=dtype))

    impact_with_season = ci.fit_causalimpact(
        test_data,
        pre_period=(test_data.index[0], test_data.index[treat_start - 1]),
        post_period=(test_data.index[treat_start], test_data.index[-1]),
        inference_options=ci.InferenceOptions(num_results=1000),
        data_options=ci.DataOptions(dtype=dtype),
        model_options=ci.ModelOptions(seasons=[
            ci.Seasons(
                num_seasons=4,
                # Ensure we can handle num_steps_per_season
                # being a tuple.
                num_steps_per_season=(2, 1, 1, 1)),
            ci.Seasons(num_seasons=7),
            ci.Seasons(
                num_seasons=6,
                # Ensure we can handle num_steps_per_season
                # being a nested tuple.
                num_steps_per_season=((2, 2, 1, 1, 1, 1), (2, 2, 1, 1, 1, 1)))
        ]))

    # Estimates of the absolute effect will be very wide when seasonality
    # is not modeled, since the variance is unexplained. When the seasonality
    # is modeled, the estimates will be much tighter.
    self.assertAlmostEqual(
        9.5,
        impact_without_season.summary["abs_effect_sd"]["average"],
        delta=1.)
    self.assertAlmostEqual(
        0.5, impact_with_season.summary["abs_effect_sd"]["average"], delta=0.1)

    self.assertSequenceEqual(
        [1000, 300, 0],
        impact_without_season.posterior_samples.seasonal_levels.shape)
    self.assertSequenceEqual(
        [1000, 300, 3],
        impact_with_season.posterior_samples.seasonal_levels.shape)


class _CausalImpactBaseTest(absltest.TestCase):

  def check_all_functions(self, impact):
    self.assertIn("Posterior Inference", ci.summary(impact))
    self.assertIn("Posterior Inference",
                  ci.summary(impact, output_format="summary"))
    self.assertIn("Analysis report", ci.summary(impact, output_format="report"))
    self.assertIn("Posterior Inference", ci.summary(impact, "summary"))
    self.assertIn("Analysis report", ci.summary(impact, "report"))
    with self.assertRaises(ValueError):
      ci.summary(impact, output_format="foo")

    self.assertIsNotNone(ci.plot(impact))


class TestDataFormats(_CausalImpactBaseTest):

  def test_missing_input(self):
    with self.assertRaises(TypeError):
      ci.fit_causalimpact()

  def test_healthy_input(self):
    data = pd.DataFrame({
        "y": np.random.randn(200),
        "x1": np.random.randn(200),
        "x2": np.random.randn(200)
    })
    impact = ci.fit_causalimpact(
        data,
        pre_period=(0, 100),
        post_period=(101, 199),
        inference_options=ci.InferenceOptions(num_results=10))
    self.assertEqual(data.shape[0], impact.series.shape[0])
    self.check_all_functions(impact)


class TestPreAndPostPeriod(_CausalImpactBaseTest):

  def test_missing_pre_period_input(self):
    data = pd.DataFrame({
        "y": np.random.randn(200),
        "x1": np.random.randn(200),
        "x2": np.random.randn(200)
    })
    data.loc[data.index[2:5], "y"] = np.nan
    impact = ci.fit_causalimpact(
        data,
        pre_period=(0, 100),
        post_period=(101, 199),
        inference_options=ci.InferenceOptions(num_results=10))
    self.assertEqual(data.shape[0], impact.series.shape[0])
    self.check_all_functions(impact)

    # Verify that the indices that have NaN outputs have NaN for everything
    # other than point predictions.
    self.assertTrue(
        np.isnan(impact.series.iloc[2:5][impact.series.columns.difference([
            "observed",
            "posterior_mean",
            "posterior_lower",
            "posterior_upper",
            "time",
            "pre_period_start",
            "pre_period_end",
            "post_period_start",
            "post_period_end",
        ])]).all(
            # Reduce across both axes.
            axis=None))

if __name__ == "__main__":
  absltest.main()
