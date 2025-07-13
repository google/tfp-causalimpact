# Copyright 2022-2023 The TFP CausalImpact Authors
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

"""Tests for standardize."""

from absl.testing import absltest

from causalimpact import standardize
import numpy as np
import pandas as pd


class StandardizeTest(absltest.TestCase):

  def testBasicDataFrame(self):
    df = pd.DataFrame({
        'x1': [4., 5., 6.],
        'x2': [100., 101., 102.],
    })
    scaler = standardize.Scaler()
    standardized_df = scaler.fit_transform(df)

    # Expect it to be standardized.
    pd.testing.assert_frame_equal(
        pd.DataFrame({
            'x1': [-1., 0., 1.],
            'x2': [-1., 0., 1.],
        }), standardized_df)

    # Expected to be back to the original.
    pd.testing.assert_frame_equal(df, scaler.inverse_transform(standardized_df))

  def testIntegerDataFrame(self):
    df = pd.DataFrame({
        'x': np.int32([4, 5, 6, 12]),
    })
    expected_df = pd.DataFrame({
        'x': np.float64([-0.7651691780042776, -0.48692584054817667,
                         -0.20868250309207573, 1.46077752164453])})
    # Expect it to be standardized, and to now be a float64 (i.e. not converted
    # back to integer) - since that is the default precision used for Integers
    # and standard deviation functions.
    pd.testing.assert_frame_equal(
        expected_df, standardize.Scaler().fit_transform(df))

  def testDataFrameWithDateIndexMaintainsIndex(self):
    df = pd.DataFrame({
        'x1': [4., 5., 6.],
        'x2': [100., 101., 102.],
    },
                      index=pd.date_range('2022-01-01', periods=3, freq='h'))
    scaler = standardize.Scaler()
    standardized_df = scaler.fit_transform(df)

    # Expect it to be standardized.
    pd.testing.assert_frame_equal(
        pd.DataFrame({
            'x1': [-1., 0., 1.],
            'x2': [-1., 0., 1.],
        },
                     index=pd.date_range('2022-01-01', periods=3, freq='h')),
        standardized_df)

    # Expected to be back to the original.
    pd.testing.assert_frame_equal(df, scaler.inverse_transform(standardized_df))

  def testNAValuesIgnoredForCalculation(self):
    df = pd.DataFrame(
        {
            'x1': [4., 5., float('nan'), 6.],
            'x2': [98., float('nan'), 102., 106.],
        },
        index=pd.date_range('2022-01-01', periods=4, freq='h'))
    scaler = standardize.Scaler()
    standardized_df = scaler.fit_transform(df)

    # Expect it to be standardized.
    pd.testing.assert_frame_equal(
        pd.DataFrame(
            {
                'x1': [-1., 0., float('nan'), 1.],
                'x2': [-1., float('nan'), 0., 1.],
            },
            index=pd.date_range('2022-01-01', periods=4, freq='h')),
        standardized_df)

  def testSubscriptAllowsPartialStandardization(self):
    df = pd.DataFrame({
        'x1': [4., 5., 6.],
        'x2': [98., 102., 106.],
    },
                      index=pd.date_range('2022-01-01', periods=3, freq='h'))
    scaler = standardize.Scaler()
    standardized_df = scaler.fit_transform(df[['x1']])

    # Expect it to be standardized.
    pd.testing.assert_frame_equal(
        pd.DataFrame({
            'x1': [-1., 0., 1.],
        },
                     index=pd.date_range('2022-01-01', periods=3, freq='h')),
        standardized_df)


if __name__ == '__main__':
  absltest.main()
