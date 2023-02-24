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

"""Provides functions for standardizing DataFrames (0 mean, 1 std)."""

import numpy as np
import pandas as pd


class NotFittedError(ValueError, AttributeError):
  """Raised if Scalar is used before fitting."""


class Scaler:
  """Standardizes dataframes to have mean 0 and standard deviation 1.

  NaNs are treated as missing values, and are ignored in `fit` and maintained
  in `transform` operations.

  API mimics scikit-learn's `StandardScaler`, except:
  - Will operate on and return a pandas DataFrame.
  - Uses an unbiased estimator for the standard deviation by default (use the
    argument `ddof=0` to recover the behavior of scikit-learn).
  """

  def __init__(self, ddof=1):
    self.ddof = ddof
    self._is_fit = False

  def fit(self, df: pd.DataFrame) -> "Scaler":
    self.mean_ = np.nanmean(df, axis=0)
    self.stddev_ = np.nanstd(df, axis=0, ddof=self.ddof)
    self._is_fit = True
    return self

  def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    if not self._is_fit:
      raise NotFittedError(
          "Must call `.fit(df)` before using Scaler to transform!")
    return pd.DataFrame(
        np.where(self.stddev_ > 0, (df - self.mean_) / self.stddev_, df),
        index=df.index,
        columns=df.columns)

  def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
    return self.fit(df).transform(df)

  def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
    if not self._is_fit:
      raise NotFittedError(
          "Must call `.fit(df)` before using Scaler to transform!")
    return (df * self.stddev_) + self.mean_
