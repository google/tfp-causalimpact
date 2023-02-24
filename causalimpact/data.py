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

"""Class for storing and preparing data for modeling."""
from typing import Optional, Text, Tuple, Union

from causalimpact import indices
from causalimpact import standardize
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp


class CausalImpactData:
  """Class for storing and preparing data for modeling.

  This class handles all of the data-related functions of CausalImpact. It
  makes sure the input data is given in an appropriate format (as a pandas
  DataFrame or convertible to pd.DataFrame), checks that the given
  outcome/feature column names exist, and splits the data into the pre-period
  and post-period based on the given treatment start time OR tuples defining
  the start/end points of the pre/post periods.

  If the pre- and post-periods do not cover the entire timespan of
  the data, the excluded portions will be used for plotting but NOT for
  fitting the model or calculating impact statistics.

  Note that the default is to standardize the pre-period data to have mean zero
  and standard deviation one. The standardization is done for each column
  separately, and then applied to the post-period data using each column's pre-
  period mean and standard deviation. The mean and standard deviation for the
  outcome column are stored so that the inferences can be back-transformed to
  the original scale.


  Attributes:
    data: Pandas DataFrame of timeseries data.
    pre_period: Start and end value in data.index for pre-intervention.
    post_period: Start and end value in data.index for post-intervention.
    outcome_column: Timeseries being modeled. Defaults to first column of
      `data`.
    feature_columns: Subset of data.columns used as covariates. `None` in case
      there are no covariates. Defaults to all non-outcome columns (or `None` if
      there are none).
    standardize_data: Boolean: Whether covariates and outcome were scaled to
      have 0 mean and 1 standard deviation.
    pre_data: Subset of `data` from `pre_period`. This is unscaled.
    after_pre_data: Subset of `data` from after the `pre_period`. The time
      between pre-period and post-period should still be forecasted to make
      accurate post-period predictions. Additionally, users are interested in
      after post-period predictions. This is unscaled.
    num_steps_forecast: Number of elements (including NaN) to forecast for,
      including the post-period and time between pre-period and post-period.
    model_pre_data: Scaled subset of `data` from `pre_period` used for fitting
      the model.
    model_after_pre_data: Scaled subset of `data` from `post_period` used for
      fitting the model.
    outcome_scaler: A `standardize.Scaler` object used to transform outcome
      data.
    feature_ts: A pd.DataFrame of the scaled data over just the feature columns.
    outcome_ts: A tfp.sts.MaskedTimeSeries instance of the outcome data from the
      `pre_period`.
  """

  def __init__(self,
               data: Union[pd.DataFrame, pd.Series],
               pre_period: Tuple[indices.InputDateType, indices.InputDateType],
               post_period: Tuple[indices.InputDateType, indices.InputDateType],
               outcome_column: Optional[Text] = None,
               standardize_data=True,
               dtype=tf.float32):
    """Constructs a `CausalImpactData` instance.

    Args:
      data: Pandas `DataFrame` containing an outcome time series and optional
        feature time series.
      pre_period: Pre-period start and end (see InputDateType).
      post_period: Post-period start and end (see InputDateType).
      outcome_column: String giving the name of the outcome column in `data`. If
        not specified, the first column in `data` is used.
      standardize_data: If covariates and output should be standardized.
      dtype: The dtype to use throughout computation.
    """
    # This is a no-op in case data is a pd.DataFrame. It is common enough to
    # pass a pd.Series that this is useful here.
    data = pd.DataFrame(data)
    self.pre_period, self.post_period = indices.parse_and_validate_date_data(
        data=data, pre_period=pre_period, post_period=post_period)
    self.data, self.outcome_column, self.feature_columns = (
        _validate_data_and_columns(data, outcome_column))
    del data  # To insure the unfiltered DataFrame is not used again.
    self.standardize_data = standardize_data
    self.pre_data = self.data.loc[(self.data.index >= self.pre_period[0])
                                  & (self.data.index <= self.pre_period[1])]
    # after_pre_data intentionally includes everything after the end of the
    # pre-period since the time between pre- and post-period needs to be
    # accounted for and we actually want to see predictions after the post
    # period.
    self.after_pre_data = self.data.loc[self.data.index > self.pre_period[1]]
    self.num_steps_forecast = len(self.after_pre_data.index)

    if self.standardize_data:
      scaler = standardize.Scaler().fit(self.pre_data)
      self.outcome_scaler = standardize.Scaler().fit(
          self.pre_data[self.outcome_column])
      self.model_pre_data = scaler.transform(self.pre_data)
      self.model_after_pre_data = scaler.transform(self.after_pre_data)
    else:
      self.outcome_scaler = None
      self.model_pre_data = self.pre_data
      self.model_after_pre_data = self.after_pre_data

    out_ts = tf.convert_to_tensor(
        self.model_pre_data[self.outcome_column], dtype=dtype)
    self.outcome_ts = tfp.sts.MaskedTimeSeries(
        time_series=out_ts, is_missing=tf.math.is_nan(out_ts))
    if self.feature_columns is not None:
      # Here we have to use the FULL time series so that the post-period
      # feature data can be used for forecasting.
      features_pre = self.model_pre_data[self.feature_columns]
      features_post = self.model_after_pre_data[self.feature_columns]
      self.feature_ts = pd.concat([features_pre, features_post], axis=0)
      self.feature_ts["intercept_"] = 1.
    else:
      self.feature_ts = None


def _validate_data_and_columns(data: pd.DataFrame,
                               outcome_column: Optional[str]):
  """Validates data and sets defaults for feature and outcome columns.

  By default, the first column of the dataframe will be used as the outcome,
  and the rest will be used as features, but these can instead be provided.

  Args:
    data: Input dataframe for analysis.
    outcome_column: Optional string to use for the outcome.

  Raises:
    KeyError: if `outcome_column` is not in the data.
    ValueError: if `outcome_column` is constant.

  Returns:
    The validated (possibly default) data, outcome column, and feature columns.
  """

  # Check outcome column -- if not specified, default is the first column.
  if outcome_column is None:
    outcome_column = data.columns[0]
  if outcome_column not in data.columns:
    raise KeyError(f"Specified `outcome_column` ({outcome_column}) not found "
                   f"in data")

  # Make sure outcome column is not constant
  if data[outcome_column].std(skipna=True, ddof=0) == 0:
    raise ValueError("Input response cannot be constant.")

  # Feature columns are all those other than the output column. Use
  # `original_column_order` to keep track of the
  # original column order, since set(data.columns) reorders the
  # columns, which leads to problems later when subsetting and transforming.
  if data.shape[1] <= 1:
    feature_columns = None
  else:
    original_column_order = data.columns
    column_differences = set(data.columns).difference([outcome_column])
    feature_columns = [
        col for col in original_column_order if col in column_differences
    ]
  data = data[[outcome_column] + (feature_columns or [])]
  if data[outcome_column].count() < 3:  # Series.count() is for non-NaN values.
    raise ValueError("Input data must have at least 3 observations.")
  if data[feature_columns or []].isna().values.any():
    raise ValueError("Input data cannot have any missing values.")
  if not data.dtypes.map(pd.api.types.is_numeric_dtype).all():
    raise ValueError("Input data must contain only numeric values.")

  return data, outcome_column, feature_columns
