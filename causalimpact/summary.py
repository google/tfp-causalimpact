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

"""Utils for printing impact summaries."""

from jinja2 import Template

summary_text = """
{% macro CI(alpha) %}{{(((1 - alpha) * 100) | string).rstrip('0').rstrip('.')}}% CI{% endmacro -%}
{% macro add_remaining_spaces(n) %}{{' ' * (19 -n)}}{% endmacro -%}
Posterior Inference {CausalImpact}
                          Average            Cumulative
Actual                    {{summary.average.actual | round(1)}}{{add_remaining_spaces(summary.average.actual | round(1) | string | length)}}{{summary.cumulative.actual | round(1)}}
Prediction (s.d.)         {{summary.average.predicted | round(1)}} ({{summary.average.predicted_sd | round(2)}}){{add_remaining_spaces(summary.average.predicted | round(1) | string | length + 3 + summary.average.predicted_sd | round(2) | string | length)}}{{summary.cumulative.predicted | round(1)}} ({{summary.cumulative.predicted_sd | round(2)}})
{{CI(alpha)}}                    [{{summary.average.predicted_lower | round(1)}}, {{summary.average.predicted_upper | round(1)}}]{{add_remaining_spaces(4 + summary.average.predicted_lower | round(1) | string | length + summary.average.predicted_upper | round(1) | string | length)}}[{{summary.cumulative.predicted_lower | round(1)}}, {{summary.cumulative.predicted_upper | round(1)}}]

Absolute effect (s.d.)    {{summary.average.abs_effect | round(1)}} ({{summary.average.abs_effect_sd | round(2)}}){{add_remaining_spaces(3 + summary.average.abs_effect | round(1) | string | length + summary.average.abs_effect_sd | round(2) | string | length)}}{{summary.cumulative.abs_effect | round(1)}} ({{summary.cumulative.abs_effect_sd | round(2)}})
{{CI(alpha)}}                    {{[summary.average.abs_effect_lower | round(1), summary.average.abs_effect_upper | round(1)] | sort}}{{add_remaining_spaces(4 + summary.average.abs_effect_lower | round(1) | string | length + summary.average.abs_effect_upper | round(1) | string | length)}}{{[summary.cumulative.abs_effect_lower | round(1), summary.cumulative.abs_effect_upper | round(1)] | sort}}

Relative effect (s.d.)    {{'{0:.1%}'.format(summary.average.rel_effect)}} ({{'{0:.1%}'.format(summary.average.rel_effect_sd|float)}}){{add_remaining_spaces(3 + '{0:.1%}'.format(summary.average.rel_effect) | length + '{0:.1%}'.format(summary.average.rel_effect_sd|float)| string | length)}}{{'{0:.1%}'.format(summary.cumulative.rel_effect)}} ({{'{0:.1%}'.format(summary.cumulative.rel_effect_sd | round(2)|float)}})
{{CI(alpha)}}                    [{{'{0:.1%}'.format([summary.average.rel_effect_lower, summary.average.rel_effect_upper]|min)}}, {{'{0:.1%}'.format([summary.average.rel_effect_upper, summary.average.rel_effect_lower]|max)}}]{{add_remaining_spaces(4 + '{0:.1%}'.format([summary.average.rel_effect_lower, summary.average.rel_effect_upper]|min)|length + '{0:.1%}'.format([summary.average.rel_effect_upper, summary.average.rel_effect_lower]|max)|length)}}[{{'{0:.1%}'.format([summary.cumulative.rel_effect_lower, summary.cumulative.rel_effect_upper]|min)}}, {{'{0:.1%}'.format([summary.cumulative.rel_effect_upper, summary.cumulative.rel_effect_lower]|max)}}]

Posterior tail-area probability p: {{p_value|round(3)}}
Posterior prob. of a causal effect: {{'{0:.2%}'.format(1 - p_value)}}

For more details run the command: summary(impact, output_format="report")
"""

report_text = """
{% set detected_sig = not (summary.average.rel_effect_lower < 0 and summary.average.rel_effect_upper > 0) -%}
{% set positive_sig = summary.average.rel_effect > 0 -%}
{% macro CI(alpha) %}{{(((1 - alpha) * 100) | string).rstrip('0').rstrip('.')}}%{% endmacro -%}
Analysis report {CausalImpact}


During the post-intervention period, the response variable had
an average value of approx. {{summary.average.actual | round(1)}}. {% if detected_sig -%}By contrast, in{% else %}In{% endif %} the absence of an
intervention, we would have expected an average response of {{summary.average.predicted | round(1)}}.
The {{CI(alpha)}} interval of this counterfactual prediction is [{{summary.average.predicted_lower | round(1)}}, {{summary.average.predicted_upper | round(1)}}].
Subtracting this prediction from the observed response yields
an estimate of the causal effect the intervention had on the
response variable. This effect is {{summary.average.abs_effect | round(1)}} with a {{CI(alpha)}} interval of
{{[summary.average.abs_effect_lower | round(1), summary.average.abs_effect_upper | round(1)] | sort}}. For a discussion of the significance of this effect,
see below.


Summing up the individual data points during the post-intervention
period (which can only sometimes be meaningfully interpreted), the
response variable had an overall value of {{summary.cumulative.actual | round(1)}}.
{% if detected_sig %}By contrast, had{% else %}Had{% endif %} the intervention not taken place, we would have expected
a sum of {{summary.cumulative.predicted| round(1)}}. The {{CI(alpha)}} interval of this prediction is {{[summary.cumulative.predicted_lower | round(1), summary.cumulative.predicted_upper | round(1)]|sort}}.


The above results are given in terms of absolute numbers. In relative
terms, the response variable showed {% if positive_sig %}an increase of +{% else %}a decrease of {% endif %}{{'{0:.1%}'.format(summary.average.rel_effect)}}. The {{CI(alpha)}}
interval of this percentage is [{{'{0:.1%}'.format([summary.average.rel_effect_lower, summary.average.rel_effect_upper] | min)}}, {{'{0:.1%}'.format([summary.average.rel_effect_upper, summary.average.rel_effect_lower]|max)}}].
{% if detected_sig and positive_sig %}

This means that the positive effect observed during the intervention
period is statistically significant and unlikely to be due to random
fluctuations. It should be noted, however, that the question of whether
this increase also bears substantive significance can only be answered
by comparing the absolute effect ({{summary.average.abs_effect | round(1)}}) to the original goal
of the underlying intervention.
{% elif detected_sig and not positive_sig %}

This means that the negative effect observed during the intervention
period is statistically significant.
If the experimenter had expected a positive effect, it is recommended
to double-check whether anomalies in the control variables may have
caused an overly optimistic expectation of what should have happened
in the response variable in the absence of the intervention.
{% elif not detected_sig and positive_sig %}

This means that, although the intervention appears to have caused a
positive effect, this effect is not statistically significant when
considering the entire post-intervention period as a whole. Individual
days or shorter stretches within the intervention period may of course
still have had a significant effect, as indicated whenever the lower
limit of the impact time series (lower plot) was above zero.
{% elif not detected_sig and not positive_sig -%}

This means that, although it may look as though the intervention has
exerted a negative effect on the response variable when considering
the intervention period as a whole, this effect is not statistically
significant and so cannot be meaningfully interpreted.
{% endif %}
{%- if not detected_sig %}

The apparent effect could be the result of random fluctuations that
are unrelated to the intervention. This is often the case when the
intervention period is very long and includes much of the time when
the effect has already worn off. It can also be the case when the
intervention period is too short to distinguish the signal from the
noise. Finally, failing to find a significant effect can happen when
there are not enough control variables or when these variables do not
correlate well with the response variable during the learning period.
{% endif %}
{%- if p_value < alpha %}

The probability of obtaining this effect by chance is very small
(Bayesian one-sided tail-area probability p = {{p_value | round(3)}}).
This means the causal effect can be considered statistically
significant.
{%- else %}

The probability of obtaining this effect by chance is p = {{'{0:.0%}'.format(p_value)}}.
This means the effect may be spurious and would generally not be
considered statistically significant.
{%- endif -%}
"""

SUMMARY_TMPL = Template(summary_text)
REPORT_TMPL = Template(report_text)


def summary(ci_model, output_format: str = "summary", alpha: float = 0.05):
  """Get summary of impact results.

  Args:
    ci_model: CausalImpact instance, after calling `.train`.
    output_format: string directing whether to print a shorter summary
      ('summary') or a long-form description ('report').
    alpha: float for alpha level to use; must be in (0, 1).

  Returns:
    Text output of summary results.
  """

  if output_format not in ["summary", "report"]:
    raise ValueError("`format` must be either 'summary' or 'report'. "
                     "Got %s" % output_format)

  if alpha <= 0. or alpha >= 1.:
    raise ValueError("`alpha` must be in (0, 1). Got %s" % alpha)

  p_value = ci_model.summary["p_value"][0]

  if output_format == "summary":
    output = SUMMARY_TMPL.render(
        summary=ci_model.summary.transpose().to_dict(),
        alpha=alpha,
        p_value=p_value)
  else:
    output = REPORT_TMPL.render(
        summary=ci_model.summary.transpose().to_dict(),
        alpha=alpha,
        p_value=p_value)

  return output
