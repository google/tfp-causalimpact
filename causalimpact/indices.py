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

"""Functions for working with indices."""

import datetime
from typing import Tuple, Union

import numpy as np
import pandas as pd

InputDateType = Union[str, int, datetime.datetime]
OutputDateType = Union[int, datetime.datetime]
InputPeriodType = Tuple[InputDateType, InputDateType]
OutputPeriodType = Tuple[OutputDateType, OutputDateType]


def parse_and_validate_date_data(
    data: pd.DataFrame,
    pre_period: InputPeriodType,
    post_period: InputPeriodType,
) -> Tuple[OutputPeriodType, OutputPeriodType]:
  """Parses and validates date arguments.

  Args:
    data: Pandas `DataFrame` containing an outcome time series and optional
      feature time series.
    pre_period: Tuple of strings giving pre-period start/end date-times (the
      date-time format should match that of data.index).
    post_period: Tuple of strings giving post-period start/end date-times (the
      date-time format should match that of data.index).

  Returns:
    A converted pre_period and post_period, which are all of the same type as
    `data.index`.
  """
  pre_period = tuple([_convert_date_to_index_type(p, data) for p in pre_period])
  post_period = tuple(
      [_convert_date_to_index_type(p, data) for p in post_period])
  pre_period, post_period = _parse_and_validate_periods(pre_period, post_period,
                                                        data)
  return pre_period, post_period


def _parse_and_validate_periods(
    pre_period: OutputPeriodType, post_period: OutputPeriodType,
    data: pd.DataFrame) -> Tuple[OutputPeriodType, OutputPeriodType]:
  """Makes sure pre- and post-periods are specified correctly.

  Args:
    pre_period: strings giving pre-period start/end date-times (the date-time
      format should match that of data.index).
    post_period: strings giving post-period start/ebd date-times (the date-time
      format should match that of data.index).
    data: dataframe with a pd.DatetimeIndex that the pre/post periods are based
      on.

  Returns:
    result: tuple of checked pre/post periods

  Raises:
    ValueError: if pre_period overlaps with post_period
                if pre-period is less than 3 time points
                if the pre/post_period dates are not given in order.
  """
  # Make sure pre/post periods are specified correctly.
  checked_pre_period = _check_period(pre_period, data)
  checked_post_period = _check_period(post_period, data)

  pre_period_dates = data.index[(data.index >= checked_pre_period[0])
                                & (data.index <= checked_pre_period[1])]

  # Make sure pre/post periods make sense.
  if checked_pre_period[1] >= checked_post_period[0]:
    raise ValueError("pre_period and post_period cannot overlap.")
  if len(pre_period_dates) < 3:
    raise ValueError("pre_period must span at least 3 time points. Got %s" %
                     len(pre_period_dates))
  if checked_pre_period[1] < checked_pre_period[0]:
    raise ValueError("pre_period last number must be bigger than its first.")
  if checked_post_period[1] < checked_post_period[0]:
    raise ValueError("post_period last number must be bigger than its first.")

  return (checked_pre_period, checked_post_period)


def _check_period(period: OutputPeriodType,
                  data: pd.DataFrame) -> OutputPeriodType:
  """Checks that periods are given appropriately.

  Args:
    period: strings giving period start/end date-times (the date-time format
      should match that of data.index).
    data: input data.

  Returns:
    period: checked period list with periods converted to datetime.

  Raises:
    ValueError: if period is not of type tuple.
                if period doesn't have two elements.
                if period has None elements.
                if period date values are not present in data.
                if period not given in order.
  """
  # Check that the dates are in order.
  if period[0] > period[1]:
    raise ValueError(f"Period end must be after period start. Got {period}")

  # Allow indices (or more likely) dates that are not aligned with the original
  # data. The period is rounded to prefer shorter periods rather than partially
  # covered periods.
  period_start_idx = data.index.get_indexer([period[0]], method="bfill")[0]
  if period_start_idx == -1:
    raise ValueError("Aligned period start not found in the index.")
  period_start = data.index[period_start_idx]

  period_end_idx = data.index.get_indexer([period[1]], method="ffill")[0]
  if period_end_idx == -1:
    raise ValueError("Aligned period end not found in the index.")
  period_end = data.index[period_end_idx]

  return (period_start, period_end)


def _convert_date_to_index_type(input_date: InputDateType,
                                data: pd.DataFrame) -> OutputDateType:
  if isinstance(input_date, str):
    return pd.to_datetime(input_date)
  elif isinstance(input_date, (int, np.integer)):
    return data.index[input_date]
  elif isinstance(input_date, datetime.datetime):
    return input_date
  else:
    raise ValueError(
        f"Expected argument to be str, int, or datetime. Got {type(input_date)}"
    )
