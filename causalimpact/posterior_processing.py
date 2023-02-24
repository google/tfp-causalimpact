# Copyright 2020-2023 The TFP CausalImpact Authors
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

"""Library for working with (results from) the posterior."""

from typing import List, Text, Tuple

from causalimpact import data as cid
import numpy as np
import pandas as pd


def calculate_trajectory_quantiles(
    trajectories: pd.DataFrame,
    column_prefix: Text = "predicted",
    quantiles: Tuple[float, float] = (0.025, 0.975)
) -> pd.DataFrame:
  """Calculates timepoint-wise quantiles of trajectories.

  This function is used to calculate timepoint-wise quantiles for both the
  posterior predictions and the cumulative predictions.

  Args:
    trajectories: pd.DataFrame of samples to take quantiles of. `trajectories`
      should have a DatetimeIndex (so rows are time points) and columns for each
      sample. The quantiles are calculated across columns (i.e. across samples
      for each time point) using the 'axis=1' argument in quantile().
    column_prefix: string giving the prefix to use for the column names of the
      quantiles; e.g. if the trajectories being passed are the cumulative ones,
      use column_prefix = "cumulative" and the returned dataframe will have
      columns "cumulative_lower" and "cumulative_upper".
    quantiles: floats in (0, 1) giving the quantiles to calculate.

  Returns:
    quantiles_df: dataframe with columns for lower/upper quantiles of
      `trajectories` and a corresponding DatetimeIndex.
  """

  column_names = [column_prefix + "_" + suffix for suffix in ["lower", "upper"]]
  # Use axis=1 to summarize across samples for each time point.
  # With two quantiles, quantile() returns a pd.DataFrame of size
  # 2 x (number of time points), so use transpose() to put time back in
  # the rows.
  quantiles_df = trajectories.quantile(q=quantiles, axis=1).transpose()
  quantiles_df.columns = column_names
  quantiles_df.index = trajectories.index

  return quantiles_df


def process_posterior_quantities(ci_data: cid.CausalImpactData,
                                 vals_to_process: np.ndarray,
                                 col_names: List[Text]) -> pd.DataFrame:
  """Process posterior quantities by undoing any scaling and reshaping.

  This function assumes that the input np.ndarray `vals_to_process` has one or
  more rows corresponding to posterior samples and columns corresponding to
  time points. IMPORTANT: this function assumes that the time points correspond
  to the full time period, pre- and post-period combined. The function does the
  following:
  * undoes any scaling, if needed
  * reshapes so that rows correspond to time points and columns to samples.
  * reformats as a pd.DataFrame with a DatetimeIndex and appropriate column
    names.

  Args:
    ci_data: an instance of a cid.CausalImpactData object.
    vals_to_process: the input array.
    col_names: list of column names to use in the output.

  Returns:
    pd.DataFrame with rows corresponding to time points and columns
    named according to `col_names`.
  """
  # If the data used for modeling were scaled, first undo the scaling.
  if ci_data.standardize_data:
    vals_to_process = ci_data.outcome_scaler.inverse_transform(vals_to_process)
  # Transpose so that rows are again time points and columns are samples.
  vals_to_process = np.transpose(vals_to_process)

  # Format as pd.DataFrame and add the appropriate DatetimeIndex. Use the union
  # of the pre/post data in case the pre/post periods don't cover the full
  # data time period.
  index = ci_data.model_pre_data.index.union(
      ci_data.model_after_pre_data.index).sort_values()
  return pd.DataFrame(vals_to_process, columns=col_names, index=index)
