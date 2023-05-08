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

"""Plotting causalimpact results."""

import altair as alt
import numpy as np
import pandas as pd
import tensorflow_probability as tfp


def plot(ci_model, **kwargs) -> alt.Chart:
  """Main plotting function.

  Args:
    ci_model: CausalImpactAnalysis object, after having called
      `fit_causalimpact`.
    **kwargs: arguments for modifying plot defaults. static_plot - whether to
      return the standard CausalImpact plot as a static plot (default) or an
      interactive plot. alpha - float for determining confidence level for
      uncertainty intervals when quantile_based_intervals=False. show_median -
      whether to draw posterior median predictions in addition to the posterior
      mean. Only applies if "median" was a specified aggregation given in
      evaluate(). use_std_intervals - whether to draw uncertainty intervals
      based on quantiles (default) or use a normal approximation based on the
      standard deviation. chart_width - integer for chart width in pixels.
      chart_height - integer for chart height in pixels. axis_title_font_size -
      integer for axis title font size. Default = 18. axis_label_font_size -
      integer for axis title font size. Default = 16. strip_title_font_size -
      integer for facet label font size. Default = 18.

  Returns:
    alt.Chart plot object
  """

  # Process kwargs.
  plot_params = {
      "static_plot": True,
      "alpha": 0.05,
      "show_median": False,
      "use_std_intervals": False,
      "chart_width": 600,
      "chart_height": 200,
      "axis_title_font_size": 18,
      "axis_label_font_size": 16,
      "strip_title_font_size": 20
  }
  if kwargs:
    for k, v in plot_params.items():
      plot_params[k] = kwargs.get(k, v)

  # Create the dataframe that will be used to create the plot.
  main_plot_df = _create_plot_df(ci_model.series, plot_params["alpha"])

  # If use_std_intervals=True, use std to draw the uncertainty intervals,
  # otherwise use the quantile-based intervals. We drop the unnecessary
  # observations instead of keeping the requested ones because dates outside of
  # the pre/post period will have NaN values for band_method.
  if plot_params["use_std_intervals"]:
    plot_df = main_plot_df.loc[main_plot_df["band_method"] != "quantile"]
  else:
    plot_df = main_plot_df.loc[main_plot_df["band_method"] != "std"]

  # Include median if requested.
  if plot_params["show_median"]:
    plot_df = plot_df.loc[plot_df["stat"] != "median"]
    plot_df["stat_pretty"] = pd.Categorical(
        plot_df["stat_pretty"], categories=["Observed", "Mean"], ordered=True)

  # Create the requested plot type.
  if plot_params["static_plot"]:
    plot_df = plot_df.loc[(plot_df["stat"] != "median")]
    plt = _draw_classic_plot(plot_df, **plot_params)
  else:
    plt = _draw_interactive_plot(plot_df, **plot_params)

  return plt


def _create_plot_df(series: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
  """Creates dataframe for use in plotting impact inferences.

  This function creates the data to be used in plotting the impact inferences.
  The plot layers lines for the observed data and predicted values along with
  uncertainty bands and then facets these depending on the scale (original,
  pointwise effect, or cumulative effect). The dataframe therefore has columns
  "time", "value", "lower", "upper", "scale", "stat" (observed/mean/median,
  if "median" was a requested aggregation). If "std" was a requested
  aggregation, the dataframe also contains "band_method" indicating whether the
  interval formed by "lower" and "upper" is calculated based on quantiles or
  a normal approximation using the predicted value +/- z_alpha/2 * std. The
  plot also includes horizontal zero lines for the pointwise and cumulative
  effect panels, so the dataframe has columns for these values too.

  Args:
    series: pd.DataFrame as output by sts_mod.evaluate(). Should be a time
      series containing observed and predicted outcome, as well as absolute and
      cumulative impact estimates.
    alpha: float determining the confidence level for std-based uncertainty
      intervals.

  Returns:
    pd.DataFrame indexed by time for creating plots. The `value` column is used
      to draw lines, while `upper` and `lower` denote the corresponding
      uncertainty bounds. The `scale` column denotes whether `value` is on the
      original, pointwise (absolute effect), or cumulative scale. The `stat`
      column denotes whether the quantity in `value` refers to the observed
      outcome or the posterior mean (or posterior median if "median" is
      in "aggregations"]). Both `scale` and `stat` have corresponding columns
      `scale_prett` and `stat_pretty` for making nice plot labels. There is also
      a `zero` column to draw a horizontal reference line for the absolute and
      cumulative effect plots.
  """
  series["time"] = series.index

  # Create dataframes for each component of the plot (lines and uncertainty
  # bands, and standard devation-based uncertainty bands if requested).
  lines_df = _create_plot_component_df(series, "lines")
  bands_df = _create_plot_component_df(series, "bands")
  if any(["std" in col for col in series.columns]):
    std_df = _create_plot_component_df(series, "std", alpha)
    bands_df = pd.concat([bands_df, std_df], axis=0, sort=True)

  # Merge the component dataframes. Use left join because lines_df will contain
  # observed data outside of the pre/post period intervals, whereas bands_df
  # will not.
  plot_df = lines_df.merge(
      bands_df,
      on=[
          "time", "scale", "pre_period_start", "pre_period_end",
          "post_period_start", "post_period_end"
      ],
      how="left")

  # Add a zero column so we can plot a zero line for the absolute and
  # cumulative scales, but set it to np.nan for the original scale so it
  # doesn't get plotted.
  plot_df["zero"] = 0
  plot_df.loc[plot_df["scale"] == "original", "zero"] = np.nan

  # Make nicer versions of the scale and stat variables for use as plot labels.
  plot_df["scale_pretty"] = [
      "Original" if m == "original" else
      "Pointwise" if m == "point_effects" else "Cumulative"
      for m in plot_df["scale"]
  ]
  plot_df["scale_pretty"] = pd.Categorical(
      plot_df["scale_pretty"],
      categories=["Original", "Pointwise", "Cumulative"],
      ordered=True)
  plot_df["stat_pretty"] = plot_df["stat"].str.capitalize()
  plot_df["stat_pretty"] = pd.Categorical(
      plot_df["stat_pretty"],
      categories=["Observed", "Mean", "Median"],
      ordered=True)

  return plot_df


def _create_plot_component_df(series: pd.DataFrame,
                              component: str,
                              alpha: float = 0.05) -> pd.DataFrame:
  """Creates plot component dataframes.

  This function takes the impact estimate time series and creates a long-form
  dataframe that can be used to plot lines or uncertainty bands for estimates
  on the original data scale, pointwise (absolute) effects, and cumulative
  effects. The dataframe is indexed by time and has columns denoting the value
  to be plotted on the y-axis, the scale that that value is on (original,
  absolute, cumulative), and the statistic that the value corresponds to (e.g.
  the upper or lower bound, the observed data, the posterior mean or median).

  Args:
    series: dataframe containing impact estimates.
    component: string for which component dataframe ("lines", "bands", "std") to
      create. Note: "bands" denotes the standard quantile-based uncertainty
      bands, while "std" denotes standard deviation-based bands.
    alpha: float determining the confidence level for std-based uncertainty
      intervals.

  Returns:
    pd.DataFrame indexed by time, with the columns `value`, `scale` (whether
      `value` is on the original, pointwise (absolute effect), or cumulative
      scale). If `component` is "lines", then the dataframe also has columns for
      `stat`, the quantity to be plotted a a line (observed, posterior mean, or
      posterior median [only if "median" is in "aggregations"]). If `component`
      is "bands" or "std", then there are columns `upper` and `lower` denoting
      the bounds of the uncertainty intervals. In this case, there is also
      a column indicating which of "bands" or "std" was used for creating the
      upper/lower bounds.
  """

  if all([x not in component for x in ["lines", "bands", "std"]]):
    raise ValueError("`component` must be one of 'lines', 'bands', or 'std'."
                     "Got %s." % component)

  # Pull out only the columns we need from `series`.
  col_stubs = [
      "time", "pre_period_start", "pre_period_end", "post_period_start",
      "post_period_end"
  ]
  if component == "lines":
    col_stubs.extend(["mean", "median", "observed"])
  elif component == "bands":
    col_stubs.extend(["lower", "upper"])
  else:  # component == "std"
    col_stubs.extend(["mean", "std"])
  cols_to_extract = [
      col for col in series.columns if any(x in col for x in col_stubs)
  ]

  # Keep the necessary columns and reshape so that `scale` and `stat` are long.
  # To do this, we collapse all of the existing columns into a single column
  # that we then split into two columns, `scale` and `stat`.
  sub_df = series[cols_to_extract].melt(
      id_vars=[
          "time", "pre_period_start", "pre_period_end", "post_period_start",
          "post_period_end"
      ],
      var_name="scale_stat",
      value_name="value")

  # `scale` is for whether `value` is on the original scale (scale of the
  # observed data), the pointwise scale, or the cumulative scale.
  stats_to_drop = "_upper|_lower|_mean|_median|_std"
  sub_df["scale"] = sub_df["scale_stat"].str.replace(
      stats_to_drop, "", regex=True)
  sub_df.loc[sub_df["scale"].str.contains("observed|posterior"),
             "scale"] = "original"

  # `stat` is for whether `value` represents the observed data or the mean or
  # median estimates.
  sub_df["stat"] = sub_df["scale_stat"].str.replace(
      "posterior_|point_effects_|cumulative_effects_", "", regex=True)
  sub_df.drop(columns=["scale_stat"], inplace=True)

  # For bands and std, reshape the data again so that "upper" and "lower" (for
  # bands) and "mean" and "std" (for std) are wide.
  # columns.
  if (component == "bands") | (component == "std"):
    sub_df = sub_df.pivot_table(
        index=[
            "time", "scale", "pre_period_start", "pre_period_end",
            "post_period_start", "post_period_end"
        ],
        columns="stat",
        values="value").reset_index()
    if component == "bands":
      sub_df["band_method"] = "quantile"
    else:
      sub_df["band_method"] = "std"

  # For std, use the posterior mean and std to create lower and upper bounds.
  if component == "std":
    z_val = tfp.distributions.Normal(
        loc=0., scale=1.).quantile(1.0 - alpha / 2.0).numpy()
    sub_df["lower"] = sub_df["mean"] - z_val * sub_df["std"]
    sub_df["upper"] = sub_df["mean"] + z_val * sub_df["std"]
    sub_df.drop(columns={"mean", "std"}, inplace=True)

  return sub_df


def _create_base_layers(plot_df: pd.DataFrame, **kwargs):
  """Create base plot layers.

  This helper function is used by both _draw_static_plot() and
  _draw_interactive_plot() to draw the four base plot layers: lines, uncertainty
  bands, and horizontal/vertical rules for zero and the treatment start,
  respectively.

  Args:
    plot_df: dataframe to use for plotting.
    **kwargs: Optional plot parameters. chart_width - integer for chart width in
      pixels. Default = 600. chart_height - integer for chart height in pixels.
      Default = 200. axis_title_font_size - integer for axis title font size.
      Default = 18. axis_label_font_size - integer for axis title font size.
      Default = 16. strip_title_font_size - integer for facet label font size.
      Default = 20.

  Returns:
    Dictionary containing the four base plot layers: lines, bands, and vertical
      and horizontal lines. NOTE that the lines, bands, and horizontal line are
      alt.Chart objects, while the vertical lines, keyed by "vlines", are
      a subdictionary of alt.Chart objects. This subdictionary is keyed by the
      date at which the vertical line is to be drawn and the values are the
      corresponding alt.Chart objects for those vertical lines.
  """

  # Base line layer.
  base_lines = alt.Chart(plot_df).mark_line().encode(
      x=alt.X("time", title="Time"),
      y=alt.Y("value:Q", scale=alt.Scale(zero=False), title="")).properties(
          width=kwargs["chart_width"], height=kwargs["chart_height"])

  # Base band layer.
  base_band = alt.Chart(plot_df).mark_area(opacity=0.3).encode(
      x=alt.X("time", title="Time"), y="upper:Q", y2="lower:Q").properties(
          width=kwargs["chart_width"], height=kwargs["chart_height"])

  # Add horizontal line at zero.
  base_hline = alt.Chart(plot_df).mark_rule().encode(y="zero")

  # Add vertical lines at the edges of the pre-period and post-period whenever
  # there is something to demarcate.
  base_vlines = {}
  # Since pre_period_end and post_period_start are columns in plot_df, we can
  # just take the values in the first row in those columns.
  pre_period_start = plot_df["pre_period_start"][0]
  pre_period_end = plot_df["pre_period_end"][0]
  post_period_start = plot_df["post_period_start"][0]
  post_period_end = plot_df["post_period_end"][0]

  # Only draw a line at the start of the pre-period if there are points before
  # it.
  if any(plot_df["time"] < pre_period_start):
    base_vlines["pre_period_start"] = alt.Chart(plot_df).mark_rule(
        strokeDash=[5, 5]).encode(
            x=alt.X("pre_period_start"), color=alt.value("grey"))
  # Only draw a line at the end of the pre-period if there are points between
  # it and the start of the post-period.
  if any((plot_df["time"] > pre_period_end)
         & (plot_df["time"] < post_period_start)):
    base_vlines["pre_period_end"] = alt.Chart(plot_df).mark_rule(
        strokeDash=[5, 5]).encode(
            x=alt.X("pre_period_end"), color=alt.value("grey"))

  # Always draw when the post-period starts.
  base_vlines["post_period_start"] = alt.Chart(plot_df).mark_rule(
      strokeDash=[5, 5]).encode(
          x=alt.X("post_period_start"), color=alt.value("grey"))

  # Only draw line at the end of the post-period if there are points after it.
  if any(plot_df["time"] > post_period_end):
    base_vlines["post_period_end"] = alt.Chart(plot_df).mark_rule(
        strokeDash=[5, 5]).encode(
            x=alt.X("post_period_end"), color=alt.value("grey"))

  # Return the base plot components.
  return {
      "lines": base_lines,
      "band": base_band,
      "hline": base_hline,
      "vlines": base_vlines
  }


def _draw_classic_plot(plot_df: pd.DataFrame, **kwargs) -> alt.Chart:
  """Draw the classic static impact plot as in the R package.

  Args:
    plot_df: dataframe to use for plotting.
    **kwargs: Optional plot parameters. chart_width - integer for chart width in
      pixels. Default = 600. chart_height - integer for chart height in pixels.
      Default = 200. axis_title_font_size - integer for axis title font size.
      Default = 18. axis_label_font_size - integer for axis title font size.
      Default = 16. strip_title_font_size - integer for facet label font size.
      Default = 20.

  Returns:
    alt.Chart object.
  """
  layers = _create_base_layers(plot_df, **kwargs)

  # Add color spec to lines.
  color_spec = alt.Color(
      "stat_pretty:N",
      legend=alt.Legend(
          title="",
          labelFontSize=kwargs["axis_label_font_size"],
          symbolSize=10 * kwargs["axis_label_font_size"]))
  layers["lines"] = layers["lines"].encode(color=color_spec)

  # Unpack the vertical rule chart objects into a list; combine with the other
  # chart layers into a tuple that can be passed to alt.layer() to create the
  # final plot.
  vlines = list(layers["vlines"].values())
  chart_layers = tuple([layers["lines"], layers["band"], layers["hline"]] +
                       vlines)
  final_plot = alt.layer(
      *chart_layers, data=plot_df).facet(
          row=alt.Row(
              "scale_pretty:N",
              sort=["Original", "Pointwise", "Cumulative"],
              title="")).resolve_scale(y="independent").configure(
                  background="white").configure_axis(
                      titleFontSize=kwargs["axis_title_font_size"],
                      labelFontSize=kwargs["axis_label_font_size"]
                  ).configure_header(
                      labelFontSize=kwargs["strip_title_font_size"])
  return final_plot


def _draw_interactive_plot(plot_df: pd.DataFrame, **kwargs) -> alt.Chart:
  """Draw interactive impact plot.

  Args:
    plot_df: dataframe to use for plotting
    **kwargs: Optional plot parameters. chart_width - integer for chart width in
      inches. Default = 15. chart_height - integer for chart height. Default =
      15. axis_title_font_size - integer for axis title font size. Default = 18.
      axis_label_font_size - integer for axis title font size. Default = 16.
      strip_title_font_size - integer for facet label font size. Default = 20.

  Returns:
    altair Chart object.
  """

  # ############################################################################
  # Create interactive selection elements.
  # ############################################################################

  # Brush for selecting a location to zoom into on x-axis.
  brush = alt.selection(type="interval", encodings=["x"])

  # Mini-chart as interactive legend to choose which stat to display.
  stat_selection = alt.selection_multi(fields=["stat_pretty"])
  selection_color = alt.condition(stat_selection,
                                  alt.Color("stat_pretty:N", legend=None),
                                  alt.value("lightgray"))
  legend = alt.Chart(plot_df).mark_point().encode(
      y=alt.Y("stat_pretty:N", axis=alt.Axis(orient="right"), title=""),
      color=selection_color).add_selection(stat_selection)

  # ############################################################################
  # Create the static top chart.
  # ############################################################################

  # The static chart is for the data on the original scale.
  static_df = plot_df.loc[plot_df["scale"] == "original"]

  # Create static layers and add the color spec to the lines layer and the
  # brush selection to the bands layer.
  static_layers = _create_base_layers(static_df, **kwargs)
  static_color_spec = alt.Color(
      "stat_pretty:N",
      legend=alt.Legend(
          title="",
          labelFontSize=kwargs["axis_label_font_size"],
          symbolSize=10 * kwargs["axis_label_font_size"]))
  static_layers["lines"] = static_layers["lines"].encode(
      color=static_color_spec)
  static_layers["band"] = static_layers["band"].add_selection(brush)

  # Combine layers of static top chart. Add the brush selection to the band
  # layer to enable selecting a date range on the x-axis for the bottom plots to
  # zoom in on (you only need to put it on one layer, so it could also have been
  # added to the lines layer).
  row_spec = alt.Row(
      "scale_pretty:N", sort=["Original", "Pointwise", "Cumulative"], title="")
  # Unpack the vertical rule chart objects into a list; combine with the other
  # chart layers into a tuple that can be passed to alt.layer to create the
  # full static plot.
  static_vlines = list(static_layers["vlines"].values())
  top_chart_layers = tuple(
      [static_layers["lines"], static_layers["band"], static_layers["hline"]] +
      static_vlines)
  top_static_plot = alt.layer(
      *top_chart_layers,
      data=static_df).facet(row=row_spec).resolve_scale(y="independent")

  # ############################################################################
  # Create the dynamic charts that will zoom upon selection on the static top
  # chart.
  # ############################################################################

  # Create dynamic layers and add interactive selections.
  dynamic_layers = _create_base_layers(plot_df, **kwargs)
  dynamic_layers["lines"] = dynamic_layers["lines"].encode(
      color=selection_color,
      x=alt.X("time", scale=alt.Scale(domain=brush), title="Time"))
  dynamic_layers["band"] = dynamic_layers["band"].encode(
      x=alt.X("time", scale=alt.Scale(domain=brush), title="Time"))

  # Add interactive selections to each of the vertical line chart objects.
  for vline_date, vline_object in dynamic_layers["vlines"].items():
    dynamic_layers["vlines"][vline_date] = vline_object.encode(
        x=alt.X(vline_date, scale=alt.Scale(domain=brush)))

  # Combine the dynamic chart layers into a tuple that can be used with
  # alt.layer() to create the full dynamic plot.
  dynamic_vlines = list(dynamic_layers["vlines"].values())
  bottom_chart_layers = tuple([
      dynamic_layers["lines"], dynamic_layers["band"], dynamic_layers["hline"]
  ] + dynamic_vlines)
  bottom_dynamic_plot = alt.layer(
      *bottom_chart_layers,
      data=plot_df).facet(row=row_spec).resolve_scale(y="independent")

  # ############################################################################
  # Create the final chart.
  # ############################################################################

  # First, vertically concatenate the static and dynamic charts, then
  # horizontally concatenate with little interactive legend chart.
  final_chart = alt.vconcat(top_static_plot, bottom_dynamic_plot)
  return (final_chart | legend).configure(background="white").configure_axis(
      titleFontSize=kwargs["axis_title_font_size"],
      labelFontSize=kwargs["axis_label_font_size"]).configure_header(
          labelFontSize=kwargs["strip_title_font_size"])
