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

"""Tests for plot.py."""

from absl.testing import absltest
from absl.testing import parameterized

import causalimpact as ci
from causalimpact.plot import _create_plot_component_df
import numpy as np
import pandas as pd


expected_classic_dict_four_vlines = {
  "facet": {
    "row": {
      "field": "scale_pretty",
      "sort": [
        "Original",
        "Pointwise",
        "Cumulative"
      ],
      "title": "",
      "type": "nominal"
    }
  },
  "spec": {
    "layer": [
      {
        "mark": {
          "type": "line"
        },
        "encoding": {
          "color": {
            "field": "stat_pretty",
            "legend": {
              "labelFontSize": 16,
              "symbolSize": 160,
              "title": ""
            },
            "type": "nominal"
          },
          "x": {
            "field": "time",
            "title": "Time",
            "type": "temporal"
          },
          "y": {
            "field": "value",
            "scale": {
              "zero": False
            },
            "title": "",
            "type": "quantitative"
          }
        }
      },
      {
        "mark": {
          "type": "area",
          "opacity": 0.3
        },
        "encoding": {
          "x": {
            "field": "time",
            "title": "Time",
            "type": "temporal"
          },
          "y": {
            "field": "upper",
            "type": "quantitative"
          },
          "y2": {
            "field": "lower"
          }
        }
      },
      {
        "mark": {
          "type": "rule"
        },
        "encoding": {
          "y": {
            "field": "zero",
            "type": "quantitative"
          }
        }
      },
      {
        "mark": {
          "type": "rule",
          "strokeDash": [
            5,
            5
          ]
        },
        "encoding": {
          "color": {
            "value": "grey"
          },
          "x": {
            "field": "pre_period_start",
            "type": "temporal"
          }
        }
      },
      {
        "mark": {
          "type": "rule",
          "strokeDash": [
            5,
            5
          ]
        },
        "encoding": {
          "color": {
            "value": "grey"
          },
          "x": {
            "field": "pre_period_end",
            "type": "temporal"
          }
        }
      },
      {
        "mark": {
          "type": "rule",
          "strokeDash": [
            5,
            5
          ]
        },
        "encoding": {
          "color": {
            "value": "grey"
          },
          "x": {
            "field": "post_period_start",
            "type": "temporal"
          }
        }
      },
      {
        "mark": {
          "type": "rule",
          "strokeDash": [
            5,
            5
          ]
        },
        "encoding": {
          "color": {
            "value": "grey"
          },
          "x": {
            "field": "post_period_end",
            "type": "temporal"
          }
        }
      }
    ],
    "height": 200,
    "width": 600
  },
  "resolve": {
    "scale": {
      "y": "independent"
    }
  }
}

expected_classic_dict_two_vlines = {
  "facet": {
    "row": {
      "field": "scale_pretty",
      "sort": [
        "Original",
        "Pointwise",
        "Cumulative"
      ],
      "title": "",
      "type": "nominal"
    }
  },
  "spec": {
    "layer": [
      {
        "mark": {
          "type": "line"
        },
        "encoding": {
          "color": {
            "field": "stat_pretty",
            "legend": {
              "labelFontSize": 16,
              "symbolSize": 160,
              "title": ""
            },
            "type": "nominal"
          },
          "x": {
            "field": "time",
            "title": "Time",
            "type": "temporal"
          },
          "y": {
            "field": "value",
            "scale": {
              "zero": False
            },
            "title": "",
            "type": "quantitative"
          }
        }
      },
      {
        "mark": {
          "type": "area",
          "opacity": 0.3
        },
        "encoding": {
          "x": {
            "field": "time",
            "title": "Time",
            "type": "temporal"
          },
          "y": {
            "field": "upper",
            "type": "quantitative"
          },
          "y2": {
            "field": "lower"
          }
        }
      },
      {
        "mark": {
          "type": "rule"
        },
        "encoding": {
          "y": {
            "field": "zero",
            "type": "quantitative"
          }
        }
      },
      {
        "mark": {
          "type": "rule",
          "strokeDash": [
            5,
            5
          ]
        },
        "encoding": {
          "color": {
            "value": "grey"
          },
          "x": {
            "field": "pre_period_end",
            "type": "temporal"
          }
        }
      },
      {
        "mark": {
          "type": "rule",
          "strokeDash": [
            5,
            5
          ]
        },
        "encoding": {
          "color": {
            "value": "grey"
          },
          "x": {
            "field": "post_period_start",
            "type": "temporal"
          }
        }
      }
    ],
    "height": 200,
    "width": 600
  },
  "resolve": {
    "scale": {
      "y": "independent"
    }
  }
}

expected_classic_dict_one_vline = {
  "facet": {
    "row": {
      "field": "scale_pretty",
      "sort": [
        "Original",
        "Pointwise",
        "Cumulative"
      ],
      "title": "",
      "type": "nominal"
    }
  },
  "spec": {
    "layer": [
      {
        "mark": {
          "type": "line"
        },
        "encoding": {
          "color": {
            "field": "stat_pretty",
            "legend": {
              "labelFontSize": 16,
              "symbolSize": 160,
              "title": ""
            },
            "type": "nominal"
          },
          "x": {
            "field": "time",
            "title": "Time",
            "type": "temporal"
          },
          "y": {
            "field": "value",
            "scale": {
              "zero": False
            },
            "title": "",
            "type": "quantitative"
          }
        }
      },
      {
        "mark": {
          "type": "area",
          "opacity": 0.3
        },
        "encoding": {
          "x": {
            "field": "time",
            "title": "Time",
            "type": "temporal"
          },
          "y": {
            "field": "upper",
            "type": "quantitative"
          },
          "y2": {
            "field": "lower"
          }
        }
      },
      {
        "mark": {
          "type": "rule"
        },
        "encoding": {
          "y": {
            "field": "zero",
            "type": "quantitative"
          }
        }
      },
      {
        "mark": {
          "type": "rule",
          "strokeDash": [
            5,
            5
          ]
        },
        "encoding": {
          "color": {
            "value": "grey"
          },
          "x": {
            "field": "post_period_start",
            "type": "temporal"
          }
        }
      }
    ],
    "height": 200,
    "width": 600
  },
  "resolve": {
    "scale": {
      "y": "independent"
    }
  }
}

expected_classic_dict_one_vline_integer_index = {
  "facet": {
    "row": {
      "field": "scale_pretty",
      "sort": [
        "Original",
        "Pointwise",
        "Cumulative"
      ],
      "title": "",
      "type": "nominal"
    }
  },
  "spec": {
    "layer": [
      {
        "mark": {
          "type": "line"
        },
        "encoding": {
          "color": {
            "field": "stat_pretty",
            "legend": {
              "labelFontSize": 16,
              "symbolSize": 160,
              "title": ""
            },
            "type": "nominal"
          },
          "x": {
            "field": "time",
            "title": "Time",
            "type": "quantitative"
          },
          "y": {
            "field": "value",
            "scale": {
              "zero": False
            },
            "title": "",
            "type": "quantitative"
          }
        }
      },
      {
        "mark": {
          "type": "area",
          "opacity": 0.3
        },
        "encoding": {
          "x": {
            "field": "time",
            "title": "Time",
            "type": "quantitative"
          },
          "y": {
            "field": "upper",
            "type": "quantitative"
          },
          "y2": {
            "field": "lower"
          }
        }
      },
      {
        "mark": {
          "type": "rule"
        },
        "encoding": {
          "y": {
            "field": "zero",
            "type": "quantitative"
          }
        }
      },
      {
        "mark": {
          "type": "rule",
          "strokeDash": [
            5,
            5
          ]
        },
        "encoding": {
          "color": {
            "value": "grey"
          },
          "x": {
            "field": "post_period_start",
            "type": "quantitative"
          }
        }
      }
    ],
    "height": 200,
    "width": 600
  },
  "resolve": {
    "scale": {
      "y": "independent"
    }
  }
}

expected_top_dict = {
  "facet": {
    "row": {
      "field": "scale_pretty",
      "sort": [
        "Original",
        "Pointwise",
        "Cumulative"
      ],
      "title": "",
      "type": "nominal"
    }
  },
  "spec": {
    "layer": [
      {
        "mark": {
          "type": "line"
        },
        "encoding": {
          "color": {
            "field": "stat_pretty",
            "legend": {
              "labelFontSize": 16,
              "symbolSize": 160,
              "title": ""
            },
            "type": "nominal"
          },
          "x": {
            "field": "time",
            "title": "Time",
            "type": "temporal"
          },
          "y": {
            "field": "value",
            "scale": {
              "zero": False
            },
            "title": "",
            "type": "quantitative"
          }
        }
      },
      {
        "mark": {
          "type": "area",
          "opacity": 0.3
        },
        "encoding": {
          "x": {
            "field": "time",
            "title": "Time",
            "type": "temporal"
          },
          "y": {
            "field": "upper",
            "type": "quantitative"
          },
          "y2": {
            "field": "lower"
          }
        },
        "name": "view_1"
      },
      {
        "mark": {
          "type": "rule"
        },
        "encoding": {
          "y": {
            "field": "zero",
            "type": "quantitative"
          }
        }
      },
      {
        "mark": {
          "type": "rule",
          "strokeDash": [
            5,
            5
          ]
        },
        "encoding": {
          "color": {
            "value": "grey"
          },
          "x": {
            "field": "pre_period_end",
            "type": "temporal"
          }
        }
      },
      {
        "mark": {
          "type": "rule",
          "strokeDash": [
            5,
            5
          ]
        },
        "encoding": {
          "color": {
            "value": "grey"
          },
          "x": {
            "field": "post_period_start",
            "type": "temporal"
          }
        }
      }
    ],
    "height": 200,
    "width": 600
  },
  "resolve": {
    "scale": {
      "y": "independent"
    }
  }
}

expected_bot_dict = {
  "facet": {
    "row": {
      "field": "scale_pretty",
      "sort": [
        "Original",
        "Pointwise",
        "Cumulative"
      ],
      "title": "",
      "type": "nominal"
    }
  },
  "spec": {
    "layer": [
      {
        "mark": {
          "type": "line"
        },
        "encoding": {
          "color": {
            "condition": {
              "param": "param_2",
              "field": "stat_pretty",
              "legend": None,
              "type": "nominal"
            },
            "value": "lightgray"
          },
          "x": {
            "field": "time",
            "scale": {
              "domain": {
                "param": "param_1"
              }
            },
            "title": "Time",
            "type": "temporal"
          },
          "y": {
            "field": "value",
            "scale": {
              "zero": False
            },
            "title": "",
            "type": "quantitative"
          }
        }
      },
      {
        "mark": {
          "type": "area",
          "opacity": 0.3
        },
        "encoding": {
          "x": {
            "field": "time",
            "scale": {
              "domain": {
                "param": "param_1"
              }
            },
            "title": "Time",
            "type": "temporal"
          },
          "y": {
            "field": "upper",
            "type": "quantitative"
          },
          "y2": {
            "field": "lower"
          }
        }
      },
      {
        "mark": {
          "type": "rule"
        },
        "encoding": {
          "y": {
            "field": "zero",
            "type": "quantitative"
          }
        }
      },
      {
        "mark": {
          "type": "rule",
          "strokeDash": [
            5,
            5
          ]
        },
        "encoding": {
          "color": {
            "value": "grey"
          },
          "x": {
            "field": "pre_period_end",
            "scale": {
              "domain": {
                "param": "param_1"
              }
            },
            "type": "temporal"
          }
        }
      },
      {
        "mark": {
          "type": "rule",
          "strokeDash": [
            5,
            5
          ]
        },
        "encoding": {
          "color": {
            "value": "grey"
          },
          "x": {
            "field": "post_period_start",
            "scale": {
              "domain": {
                "param": "param_1"
              }
            },
            "type": "temporal"
          }
        }
      }
    ],
    "height": 200,
    "width": 600
  },
  "resolve": {
    "scale": {
      "y": "independent"
    }
  }
}

expected_legend_dict = {
  "mark": {
    "type": "point"
  },
  "encoding": {
    "color": {
      "condition": {
        "param": "param_2",
        "field": "stat_pretty",
        "legend": None,
        "type": "nominal"
      },
      "value": "lightgray"
    },
    "y": {
      "axis": {
        "orient": "right"
      },
      "field": "stat_pretty",
      "title": "",
      "type": "nominal"
    }
  },
  "name": "view_2"
}


class PlotTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(PlotTest, cls).setUpClass()

    # Create a fake impact time series dataframe.
    n_time_points = 10
    time_index = pd.date_range("2018-01-01", periods=n_time_points, freq="D")
    observed = pd.Series([0, 1, 2, 3, 4, 8, 9, 10, 11, 12])
    posterior_mean = pd.Series(list(range(n_time_points)))
    posterior_median = pd.Series(list(x + 0.1 for x in range(n_time_points)))
    posterior_lower = pd.Series(list(x - 0.2 for x in range(n_time_points)))
    posterior_upper = pd.Series(list(x + 0.2 for x in range(n_time_points)))
    posterior_std = pd.Series(np.repeat(0.1, repeats=n_time_points))
    point_effects_mean = pd.Series([0, 0, 0, 0, 0, 3, 3, 3, 3, 3])
    point_effects_lower = pd.Series([0, 0, 0, 0, 0, 2.8, 2.8, 2.8, 2.8, 2.8])
    point_effects_upper = pd.Series([0, 0, 0, 0, 0, 3.2, 3.2, 3.2, 3.2, 3.2])
    point_effects_std = pd.Series(np.repeat(0.1, repeats=n_time_points))
    cumulative_effects_mean = pd.Series([0, 0, 0, 0, 0, 3, 6, 9, 12, 15])
    cumulative_effects_lower = pd.Series(
        [0, 0, 0, 0, 0, 2.8, 5.6, 8.4, 11.2, 14])
    cumulative_effects_upper = pd.Series(
        [0, 0, 0, 0, 0, 3.2, 6.4, 9.6, 12.8, 16])
    cumulative_effects_std = pd.Series(np.repeat(0.1, repeats=n_time_points))
    df = pd.concat([
        observed, posterior_mean, posterior_median, posterior_lower,
        posterior_upper, posterior_std, point_effects_mean, point_effects_lower,
        point_effects_upper, point_effects_std, cumulative_effects_mean,
        cumulative_effects_lower, cumulative_effects_upper,
        cumulative_effects_std
    ],
                   axis=1)
    df.columns = [
        "observed", "posterior_mean", "posterior_median", "posterior_lower",
        "posterior_upper", "posterior_std", "point_effects_mean",
        "point_effects_lower", "point_effects_upper", "point_effects_std",
        "cumulative_effects_mean", "cumulative_effects_lower",
        "cumulative_effects_upper", "cumulative_effects_std"
    ]
    df.index = time_index
    series_1 = df.copy()
    series_1["pre_period_start"] = series_1.index[0]
    series_1["pre_period_end"] = series_1.index[3]
    series_1["post_period_start"] = series_1.index[4]
    series_1["post_period_end"] = series_1.index[-1]
    cls.ci_data_1 = ci.CausalImpactAnalysis(
        series=series_1, summary=pd.DataFrame(), posterior_samples=[])
    series_2 = df.copy()
    series_2["pre_period_start"] = series_2.index[0]
    series_2["pre_period_end"] = series_2.index[3]
    series_2["post_period_start"] = series_2.index[6]
    series_2["post_period_end"] = series_2.index[-1]
    cls.ci_data_2 = ci.CausalImpactAnalysis(
        series=series_2, summary=pd.DataFrame(), posterior_samples=[])

    # This is constructed to have start and end of pre-period and post-period
    # drawn by having a point before/intbetween/after each of them.
    series_4 = df.copy()
    series_4["pre_period_start"] = series_4.index[1]
    series_4["pre_period_end"] = series_4.index[3]
    series_4["post_period_start"] = series_4.index[6]
    series_4["post_period_end"] = series_4.index[-2]
    cls.ci_data_4 = ci.CausalImpactAnalysis(
        series=series_4, summary=pd.DataFrame(), posterior_samples=[])

    series_integer_index = df.copy()
    series_integer_index.index = pd.RangeIndex(stop=n_time_points)
    series_integer_index["pre_period_start"] = series_integer_index.index[0]
    series_integer_index["pre_period_end"] = series_integer_index.index[3]
    series_integer_index["post_period_start"] = series_integer_index.index[4]
    series_integer_index["post_period_end"] = series_integer_index.index[-1]
    cls.ci_data_integer_index = ci.CausalImpactAnalysis(
        series=series_integer_index,
        summary=pd.DataFrame(),
        posterior_samples=[])

  def testCreatePlotComponentDF_lines(self):
    lines_df = _create_plot_component_df(
        self.ci_data_1.series, component="lines")
    expected_cols = [
        "time", "post_period_start", "post_period_end", "pre_period_start",
        "pre_period_end", "value", "scale", "stat"
    ]
    self.assertSetEqual(set(expected_cols), set(lines_df.columns))

  @parameterized.named_parameters([
      {
          "testcase_name": "quantile",
          "component": "bands",
          "method": "quantile"
      },
      {
          "testcase_name": "std",
          "component": "std",
          "method": "std"
      },
  ])
  def testCreatePlotComponentDF_bands(self, component, method):
    bands_df = _create_plot_component_df(
        self.ci_data_2.series, component=component)
    expected_cols = [
        "time", "post_period_start", "post_period_end", "pre_period_start",
        "pre_period_end", "lower", "upper", "scale", "band_method"
    ]
    self.assertSetEqual(set(expected_cols), set(bands_df.columns))

    # `bands_df` has a column specifying which method ("quantile" or "std") was
    # used to create the uncertainty intervals; here we make sure the correct
    # method is given.
    self.assertEqual(bands_df["band_method"].unique(), method)

  def testPlotMatplotlib(self):
    # Create plot object and use to_dict() to convert the plot components into
    # a more easily queried dict.
    fig = ci.plot(self.ci_data_1, backend="matplotlib")
    self.assertIsNotNone(fig)

  def testClassicPlot_one_vline(self):
    # Create plot object and use to_dict() to convert the plot components into
    # a more easily queried dict.
    classic_plot_dict = ci.plot(self.ci_data_1).to_dict()

    # Check the important elements: that facets are mapped to the correct
    # variable, the plot layers (lines, bands, etc) have the correct specs, and
    # the scale resolutions are the same.
    keys_to_test = ["facet", "spec", "resolve"]
    classic_plot_dict = {
        k: v for k, v in classic_plot_dict.items() if k in keys_to_test
    }

    self.assertDictEqual(classic_plot_dict, expected_classic_dict_one_vline)

  def testClassicPlot_one_vline_integer_index(self):
    # Create plot object and use to_dict() to convert the plot components into
    # a more easily queried dict.
    classic_plot_dict = ci.plot(self.ci_data_integer_index).to_dict()

    # Check the important elements: that facets are mapped to the correct
    # variable, the plot layers (lines, bands, etc) have the correct specs, and
    # the scale resolutions are the same.
    keys_to_test = ["facet", "spec", "resolve"]
    classic_plot_dict = {
        k: v for k, v in classic_plot_dict.items() if k in keys_to_test
    }
    self.assertDictEqual(classic_plot_dict,
                         expected_classic_dict_one_vline_integer_index)

  def testClassicPlot_two_vlines(self):
    # Create plot object and use to_dict() to convert the plot components into
    # a more easily queried dict.
    classic_plot_dict = ci.plot(self.ci_data_2).to_dict()

    # Check the important elements: that facets are mapped to the correct
    # variable, the plot layers (lines, bands, etc) have the correct specs, and
    # the scale resolutions are the same.
    keys_to_test = ["facet", "spec", "resolve"]
    classic_plot_dict = {
        k: v for k, v in classic_plot_dict.items() if k in keys_to_test
    }
    self.assertDictEqual(classic_plot_dict, expected_classic_dict_two_vlines)

  def testClassicPlot_four_vlines(self):
    # Create plot object and use to_dict() to convert the plot components into
    # a more easily queried dict.
    classic_plot_dict = ci.plot(self.ci_data_4).to_dict()

    # Check the important elements: that facets are mapped to the correct
    # variable, the plot layers (lines, bands, etc) have the correct specs, and
    # the scale resolutions are the same.
    keys_to_test = ["facet", "spec", "resolve"]
    classic_plot_dict = {
        k: v for k, v in classic_plot_dict.items() if k in keys_to_test
    }
    self.assertDictEqual(classic_plot_dict, expected_classic_dict_four_vlines)

  def testInteractivePlot(self):

    # Create plot object and use to_dict() to convert the plot components into
    # a more easily queried dict.
    interactive_plot_dict = ci.plot(self.ci_data_2, static_plot=False).to_dict()

    # Extract the relevant subdictionaries. Remove the `data` elements since
    # we don't need to check those.
    top_plot_dict = interactive_plot_dict["hconcat"][0]["vconcat"][0]
    bot_plot_dict = interactive_plot_dict["hconcat"][0]["vconcat"][1]
    legend_dict = interactive_plot_dict["hconcat"][1]
    del top_plot_dict["data"]
    del bot_plot_dict["data"]
    del legend_dict["data"]

    # Check that: 1) facets are mapped to the correct variable, 2) plot layers
    # (lines, bands, etc) have the correct specs for both the top panel and the
    # bottom three dynamic plot panels, and 3) the interactive legend is
    # correctly specified.
    self.assertDictEqual(top_plot_dict, expected_top_dict)
    self.assertDictEqual(bot_plot_dict, expected_bot_dict)
    self.assertDictEqual(legend_dict, expected_legend_dict)


if __name__ == "__main__":
  absltest.main()
