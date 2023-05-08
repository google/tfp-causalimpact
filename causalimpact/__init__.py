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

"""Import commonly used modules and classes so user doesn't have to."""

__version__ = "0.2.0"

import os

# TF 2.12 currently generates a huge number of useless log.info messages
# directly from C++, which cannot be filtered using standard Python logging
# tools -- see https://github.com/tensorflow/tensorflow/issues/59779 .
# To improve the TFP CausalImpact user experience, we suppress these messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# pylint: disable=g-import-not-at-top
from causalimpact.causalimpact_lib import CausalImpactAnalysis
from causalimpact.causalimpact_lib import DataOptions
from causalimpact.causalimpact_lib import fit_causalimpact
from causalimpact.causalimpact_lib import InferenceOptions
from causalimpact.causalimpact_lib import ModelOptions
from causalimpact.causalimpact_lib import Seasons
from causalimpact.indices import InputDateType
from causalimpact.plot import plot
from causalimpact.summary import summary
