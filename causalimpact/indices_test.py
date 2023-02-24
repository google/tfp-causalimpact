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

"""Tests for indices.py."""

import os
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized

from causalimpact import indices

import numpy as np
import pandas as pd


CURR_PATH = os.path.dirname(__file__)
TEST_DATA = os.path.join(CURR_PATH, "testdata", "data.csv")


class WrapperTest(parameterized.TestCase):
  """Tests for indices."""

  @classmethod
  def setUpClass(cls):
    super(WrapperTest, cls).setUpClass()

    # Read in sample data and set column "t" as the index
    with open(TEST_DATA, "r") as fl:
      df = pd.read_csv(fl)
    df = df.set_index(pd.to_datetime(df["t"]))
    df.drop(columns=["t"], inplace=True)

    cls.data = df

  @parameterized.named_parameters([
      {
          "testcase_name": "dates_before_the_data",
          "period": (pd.to_datetime("2016-01-20 22:41:20"),
                     pd.to_datetime("2016-01-21 22:41:20")),
          "error_string": "Aligned period end not found in the index."
      },
      {
          "testcase_name": "dates_after_the_data",
          "period": (pd.to_datetime("2022-01-20 22:41:20"),
                     pd.to_datetime("2022-01-21 22:41:20")),
          "error_string": "Aligned period start not found in the index."
      },
      {
          "testcase_name": "dates_not_in_order",
          "period": (pd.to_datetime("2016-02-20 22:41:50"),
                     pd.to_datetime("2016-02-20 22:41:20")),
          "error_string": "Period end must be after period start. "
      },
  ])
  def testCheckPeriods(self, period: Tuple[str], error_string: str):
    with self.assertRaisesRegex(ValueError, error_string):
      indices._check_period(period, self.data)

  @parameterized.named_parameters([
      {
          "testcase_name": "overlapping_periods",
          "pre_period": (pd.to_datetime("2016-02-20 22:41:20"),
                         pd.to_datetime("2016-02-20 22:41:50")),
          "post_period": (pd.to_datetime("2016-02-20 22:41:40"),
                          pd.to_datetime("2016-02-20 22:41:50")),
          "error_string": "pre_period and post_period cannot overlap."
      },
      {
          "testcase_name": "pre_period_too_short",
          "pre_period": (pd.to_datetime("2016-02-20 22:41:20"),
                         pd.to_datetime("2016-02-20 22:41:30")),
          "post_period": (pd.to_datetime("2016-02-20 22:41:40"),
                          pd.to_datetime("2016-02-20 22:41:50")),
          "error_string": "pre_period must span at least 3 time points."
      },
  ])
  def testParseAndValidatePeriods(self, pre_period: Tuple[str],
                                  post_period: Tuple[str], error_string: str):
    with self.assertRaisesRegex(ValueError, error_string):
      indices._parse_and_validate_periods(  # pylint: disable=protected-access
          pre_period, post_period, self.data)

  @parameterized.named_parameters([
      {
          "testcase_name": "_integer_periods",
          "pre_period": (0, 10),
          "post_period": (11, 90)
      },
      {
          "testcase_name": "_string_periods",
          "pre_period": ("2016-02-20 22:41:20", "2016-02-20 22:43:00"),
          "post_period": ("2016-02-20 22:43:10", "2016-02-20 22:56:20"),
      },
      {
          "testcase_name":
              "_datetime_periods",
          "pre_period": (pd.to_datetime("2016-02-20 22:41:20"),
                         pd.to_datetime("2016-02-20 22:43:00")),
          "post_period": (pd.to_datetime("2016-02-20 22:43:10"),
                          pd.to_datetime("2016-02-20 22:56:20")),
      },
  ])
  def testParseAndValidateData(self, pre_period, post_period):
    """Test different datetime arguments are correctly handled."""
    pre_period, post_period = indices.parse_and_validate_date_data(
        data=self.data, pre_period=pre_period, post_period=post_period)

    self.assertTupleEqual(pre_period, (pd.to_datetime("2016-02-20 22:41:20"),
                                       pd.to_datetime("2016-02-20 22:43:00")))
    self.assertTupleEqual(post_period, (pd.to_datetime("2016-02-20 22:43:10"),
                                        pd.to_datetime("2016-02-20 22:56:20")))

  @parameterized.named_parameters([{
      "testcase_name": "_integer_periods",
      "pre_period": (0, 10),
      "post_period": (11, 90)
  }])
  def testParseAndValidateNonDatetimeData(self, pre_period, post_period):
    """Test in cases data has an integer index."""
    data = self.data.copy()
    data.index = np.arange(data.shape[0])
    pre_period, post_period = indices.parse_and_validate_date_data(
        data=data, pre_period=pre_period, post_period=post_period)

    self.assertTupleEqual(pre_period, (0, 10))
    self.assertTupleEqual(post_period, (11, 90))


if __name__ == "__main__":
  absltest.main()
