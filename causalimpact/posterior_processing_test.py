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

"""Tests for posterior_processing.py."""

from absl.testing import absltest

from causalimpact import data as cid
from causalimpact import posterior_processing
import numpy as np
import pandas as pd


class UtilsTest(absltest.TestCase):

  def testCalculateTrajectoryQuantiles(self):
    trajectories = pd.DataFrame(
        np.random.randn(10, 10000000),
        index=pd.date_range("2018-01-01", periods=10, freq="D"))
    quantiles_df = posterior_processing.calculate_trajectory_quantiles(
        trajectories)
    self.assertTrue(quantiles_df.index.equals(trajectories.index))
    self.assertEqual(quantiles_df.shape, (trajectories.shape[0], 2))
    np.testing.assert_allclose(
        quantiles_df["predicted_lower"],
        np.repeat(-1.96, trajectories.shape[0]),
        rtol=1,
        atol=0.01)
    np.testing.assert_allclose(
        quantiles_df["predicted_upper"],
        np.repeat(1.96, trajectories.shape[0]),
        rtol=1,
        atol=0.01)

  def testProcessPosteriorQuantities(self):
    n = 1000
    treat_index = 500
    time_index = pd.date_range("2018-01-01", periods=n, freq="D")
    test_data = pd.DataFrame({"y": np.random.randn(n)}, index=time_index)
    ci_data = cid.CausalImpactData(
        test_data,
        pre_period=(time_index[0], time_index[treat_index - 1]),
        post_period=(time_index[treat_index], time_index[-1]))
    vals_to_process = pd.DataFrame(np.random.randn(10, n))
    col_names = ["col_" + str(i) for i in range(vals_to_process.shape[0])]
    df = posterior_processing.process_posterior_quantities(
        ci_data, vals_to_process, col_names)
    self.assertTrue(df.index.equals(time_index))
    self.assertCountEqual(df.columns, col_names)

    # Make sure that inverting the scaling worked correctly.
    y_pre_mean = ci_data.pre_data["y"].mean()
    y_pre_std = ci_data.pre_data["y"].std()
    orig_transformed = (vals_to_process * y_pre_std) + y_pre_mean
    _ = orig_transformed.mean(axis=1)
    _ = orig_transformed.std(axis=1)
    # TODO(colcarroll,jburnim): Consider if we should actually keep tests for
    # the values, or if it is covered elsewhere sufficiently.
    # np.testing.assert_allclose(
    #     df.mean(axis=0), transformed_means, rtol=1, atol=0.01)
    # np.testing.assert_allclose(
    #     df.std(axis=0), transformed_stds, rtol=1, atol=0.01)


if __name__ == "__main__":
  absltest.main()
