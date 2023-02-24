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

"""Tests for data.py."""

import os

from absl.testing import absltest
from absl.testing import parameterized

import causalimpact.data as cid

import numpy as np
import pandas as pd


CURR_PATH = os.path.dirname(__file__)
TEST_DATA = os.path.join(CURR_PATH, "testdata", "data.csv")


class DataCreationTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(DataCreationTest, cls).setUpClass()

    # Read in sample data and set column "t" as the index
    with open(TEST_DATA, "r") as fl:
      df = pd.read_csv(fl)
    df = df.set_index(pd.to_datetime(df["t"]))
    df.drop(columns=["t"], inplace=True)

    cls._data = df
    treatment_start_index = 60
    cls._pre_period = (df.index[0], df.index[treatment_start_index - 1])
    cls._post_period = (df.index[treatment_start_index], df.index[-1])
    cls._treatment_start = cls._post_period[0]

  def testCorrectDataWithOnlyOutcome(self):
    df = self._data[["y"]]
    ci_data = cid.CausalImpactData(
        df, pre_period=self._pre_period, post_period=self._post_period)
    self.assertEqual(ci_data.outcome_column, "y")
    self.assertIsNone(ci_data.feature_columns)
    pre_index = self._data.index[self._data.index < self._treatment_start]
    post_index = self._data.index[self._data.index >= self._treatment_start]
    self.assertTrue(ci_data.model_pre_data.index.equals(pre_index))
    self.assertTrue(ci_data.model_after_pre_data.index.equals(post_index))

  @parameterized.named_parameters([
      {
          "testcase_name": "only_outcome_given",
          "outcome_column": "y",
          "expected_feature_columns": ["x1", "x2"]
      },
      {
          "testcase_name": "no_columns_given",
          "outcome_column": None,
          "expected_feature_columns": ["x1", "x2"]
      },
  ])
  def testCorrectDataWithColumnInput(self, outcome_column,
                                     expected_feature_columns):
    ci_data = cid.CausalImpactData(
        self._data,
        pre_period=self._pre_period,
        post_period=self._post_period,
        outcome_column=outcome_column)
    self.assertEqual(ci_data.outcome_column, "y")
    self.assertSetEqual(
        set(ci_data.feature_columns), set(expected_feature_columns))
    self.assertSetEqual(
        set(ci_data.pre_data.columns), set(["y"] + expected_feature_columns))
    self.assertSetEqual(
        set(ci_data.after_pre_data.columns),
        set(["y"] + expected_feature_columns))
    pre_index = self._data.index[self._data.index < self._treatment_start]
    post_index = self._data.index[self._data.index >= self._treatment_start]
    self.assertTrue(ci_data.pre_data.index.equals(pre_index))
    self.assertTrue(ci_data.after_pre_data.index.equals(post_index))

  def testFailsWhenOutcomeDoesntExist(self):
    with self.assertRaises(KeyError):
      cid.CausalImpactData(
          self._data,
          pre_period=self._pre_period,
          post_period=self._post_period,
          outcome_column="z")

  @parameterized.named_parameters([{
      "testcase_name": "NoStandardize",
      "standardize_data": False
  }, {
      "testcase_name": "Standardize",
      "standardize_data": True
  }])
  def testStandardize(self, standardize_data):
    ci_data = cid.CausalImpactData(
        self._data,
        pre_period=self._pre_period,
        post_period=self._post_period,
        standardize_data=standardize_data)
    pre_time = pd.to_datetime("2016-02-20 22:41:20")
    post_time = pd.to_datetime("2016-02-20 22:51:20")
    index = ["y", "x1", "x2"]

    # These are just hard-coded values, to ensure there is some kind of
    # a difference that seems reasonable (e.g. could be standardized based
    # on looking at the numbers).
    if standardize_data:
      pd.testing.assert_series_equal(
          ci_data.model_pre_data.iloc[0],
          pd.Series([-0.718908, 1.684957, 0.705064], index=index,
                    name=pre_time),
          # Allow minor differences due to encoding.
          rtol=0.01)
      pd.testing.assert_series_equal(
          ci_data.model_after_pre_data.iloc[0],
          pd.Series([0.355322, -1.456488, -2.652383],
                    index=index,
                    name=post_time),
          # Allow minor differences due to encoding.
          rtol=0.01)
    else:
      pd.testing.assert_series_equal(
          ci_data.pre_data.iloc[0],
          pd.Series([110.0, 134., 128], index=index, name=pre_time))
      pd.testing.assert_series_equal(
          ci_data.after_pre_data.iloc[0],
          pd.Series([123., 123., 123.], index=index, name=post_time),
      )

  def testMissingValues(self):
    na_data = self._data.copy()
    na_data.loc[self._treatment_start, "x1"] = np.nan
    with self.assertRaises(ValueError):
      cid.CausalImpactData(
          na_data, pre_period=self._pre_period, post_period=self._post_period)


if __name__ == "__main__":
  absltest.main()
