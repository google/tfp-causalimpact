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

"""Tests for summary.py."""

import os

from absl.testing import absltest
from absl.testing import parameterized

import causalimpact as ci
import numpy as np
import pandas as pd


CURR_PATH = os.path.dirname(__file__)
TEST_PATH = os.path.join(CURR_PATH, "testdata")


def summary_data():
  """Create fake summary data."""
  data = [
      [5.343, 10.343],
      [4.343, 9.343],
      [3.343, 8.343],
      [6.343, 9.343],
      [0.001, 0.100],
      [3.343, 10.343],
      [2.343, 4.343],
      [6.343, 9.343],
      [0.001, 0.100],
      [0.123, 0.233],
      [0.143, 0.133],
      [0.343, 0.333],
      [0.001, 0.100],
  ]

  # Transpose data to get in right shape.
  summary = pd.DataFrame(
      np.array(data).T.tolist(),
      columns=[
          "actual", "predicted", "predicted_lower", "predicted_upper",
          "predicted_sd", "abs_effect", "abs_effect_lower", "abs_effect_upper",
          "abs_effect_sd", "rel_effect", "rel_effect_lower", "rel_effect_upper",
          "rel_effect_sd"
      ],
      index=["average", "cumulative"])
  return ci.CausalImpactAnalysis(
      series=pd.DataFrame(), summary=summary, posterior_samples=[])


class SummaryTest(parameterized.TestCase):

  @parameterized.named_parameters([
      {
          "testcase_name": "summary1",
          "p_value": 0.5,
          "rel_col_values": [0.41, -0.30, 0.30],
          "file_num": 1
      },
      {
          "testcase_name": "summary2",
          "p_value": 0.05,
          "rel_col_values": [0.41, 0.434, 0.234],
          "file_num": 2
      },
      {
          "testcase_name": "summary3",
          "p_value": 0.5,
          "rel_col_values": [-0.343, -0.434, 0.234],
          "file_num": 3
      },
      {
          "testcase_name": "summary4",
          "p_value": 0.05,
          "rel_col_values": [-0.343, -0.434, -0.234],
          "file_num": 4
      },
  ])
  def testReport(self, p_value, rel_col_values, file_num):
    ci_data = summary_data()
    ci_data.summary["p_value"] = p_value
    rel_cols = ["rel_effect", "rel_effect_lower", "rel_effect_upper"]
    ci_data.summary.loc["average", rel_cols] = rel_col_values
    output = ci.summary(ci_data, output_format="report", alpha=0.1).strip()
    with open(
        os.path.join(TEST_PATH, "test_report_text_" + str(file_num) + ".txt"),
        "r") as f:
      expected = f.read().strip()

    self.assertEqual(output, expected)

  def testSummary(self):
    ci_data = summary_data()
    ci_data.summary["p_value"] = 0.459329
    output = ci.summary(ci_data, output_format="summary", alpha=0.1).strip()
    with open(os.path.join(TEST_PATH, "test_summary_output.txt"), "r") as f:
      expected = f.read().strip()

    self.assertEqual(output, expected)


if __name__ == "__main__":
  absltest.main()
