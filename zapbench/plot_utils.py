# Copyright 2024 The Google Research Authors.
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

"""Plotting utilities."""

import collections
from typing import Any, Sequence

import altair as alt
import pandas as pd
from zapbench import constants

Chart = alt.vegalite.v4.api.Chart
FacetChart = alt.vegalite.v4.api.FacetChart


METHOD_TO_HEX_COLOR = collections.OrderedDict({
    'linear': '#9334E6',
    'tide': '#E52592',
    'tsmixer': '#E8710A',
    'time-mix': '#F9AB00',
    'unet': '#7CB342',
})

CONTEXT_LENGTH_TO_LABEL = collections.OrderedDict({
    '4': 'short context',
    '256': 'long context',
})

CONTEXT_LABEL_TO_SHAPE = collections.OrderedDict({
    'short context': 'circle',
    'long context': 'square',
})


def register_and_enable_custom_theme(
    font_size: int = 28, font_family: str = 'Arial'
):
  """Registers and enables custom theme."""
  alt.themes.register(
      'custom',
      lambda: {
          'config': {
              'axis': {
                  'labelFont': font_family,
                  'labelFontSize': font_size * 0.50,
                  'titleFont': font_family,
                  'titleFontSize': font_size * 0.65,
              },
              'header': {
                  'labelFont': font_family,
                  'labelFontSize': font_size * 0.65,
                  'titleFont': font_family,
                  'titleFontSize': font_size,
              },
              'legend': {
                  'labelFont': font_family,
                  'labelFontSize': font_size * 0.65,
                  'titleFont': font_family,
                  'titleFontSize': font_size * 0.65,
              },
              'title': {
                  'font': font_family,
                  'fontSize': font_size,
              },
          }
      },
  )
  alt.themes.enable('custom')


def get_ordered_elements_and_values(
    elements: Sequence[Any],
    ordered_mapping: collections.OrderedDict[str, Any],
    default_value: Any,
    excluded_elements: Sequence[Any] = (),
) -> tuple[Sequence[Any], Sequence[Any]]:
  """Gets a list of ordered elements and their corresponding values.

  Adheres to the ordering of keys in `ordered_mapping`. If an element is found
  in `ordered_mapping`, its associated value is used, otherwise `default_value`.

  Args:
    elements: Sequence of elements to be ordered.
    ordered_mapping: Ordered dictionary mapping elements to values.
    default_value: Default value to use for elements not in `ordered_mapping`.
    excluded_elements: Sequence of elements to exclude from the output.

  Returns:
    Tuple containing two lists, ordered elements, and corresponding values.
  """
  values, ordered_elements, seen_elements = [], [], set()

  for element in ordered_mapping:
    if element in elements and element not in excluded_elements:
      ordered_elements.append(element)
      values.append(ordered_mapping[element])
      seen_elements.add(element)

  for element in elements:
    if element not in seen_elements and element not in excluded_elements:
      ordered_elements.append(element)
      values.append(default_value)

  return ordered_elements, values


def create_facet_chart(
    data: Any,
    charts: Any,
    column: str | None = None,
    column_header_kwargs: dict[str, Any] | None = None,
    column_labels: bool = True,
    column_sort: Sequence[str] | None = None,
    column_title: str = '',
    row: str | None = None,
    row_header_kwargs: dict[str, Any] | None = None,
    row_labels: bool = True,
    row_sort: Sequence[str] | None = None,
    row_title: str = '',
    label_orientation: str = 'top',
    title: str = '',
    title_orientation: str = 'top',
    facet_spacing: int = 12,
    facet_width: int | None = None,
    facet_height: int | None = None,
) -> FacetChart:
  """Creates a facet chart with multiple layers."""
  if row_header_kwargs is None:
    row_header_kwargs = dict(labelOrient='left')

  chart = alt.layer(*charts, data=data)
  chart = (
      chart.properties(width=facet_width) if facet_width is not None else chart
  )
  chart = (
      chart.properties(height=facet_height)
      if facet_height is not None
      else chart
  )

  facet_kwargs = {'spacing': facet_spacing}
  if row is not None:
    facet_kwargs['row'] = alt.Row(
        row,
        header=alt.Header(
            labels=row_labels,
            **({} if row_header_kwargs is None else row_header_kwargs),
        ),
        sort=row_sort,
        title=row_title,
    )
  if column is not None:
    facet_kwargs['column'] = alt.Column(
        column,
        header=alt.Header(
            labels=column_labels,
            **({} if column_header_kwargs is None else column_header_kwargs),
        ),
        sort=column_sort,
        title=column_title,
    )
  if row is not None or column is not None:
    chart = chart.facet(**facet_kwargs, title=title)

  return chart.configure_header(
      titleOrient=title_orientation, labelOrient=label_orientation
  )


def _validate_data_for_plotting(
    data: pd.DataFrame,
    required_columns: Sequence[str] = ('context', 'method', 'steps_ahead'),
):
  """Validates DataFrame for plotting."""
  if not all(col in data.columns for col in required_columns):
    raise ValueError(
        f'Input DataFrame must contain columns: {required_columns}'
    )

  if 'context' in required_columns:
    expected_contexts_lengths = list(CONTEXT_LENGTH_TO_LABEL.keys())
    for actual in set(data['context'].unique()):
      if str(actual) not in expected_contexts_lengths:
        raise ValueError(
            f"Expected 'context' values to be in {expected_contexts_lengths}, "
            f'but found {actual}'
        )


def _format_steps_ahead(steps: float) -> str:
  """Formats the 'steps_ahead' value for display."""
  steps_str = str(int(steps))  # Convert to integer string and remove decimal
  return (
      f'{steps_str} step ahead'
      if steps_str == '1'
      else f'{steps_str} steps ahead'
  )


def _sort_steps_ahead(strings: Sequence[str]) -> Sequence[str]:
  """Sorts the 'steps_ahead' values."""
  return sorted(
      strings,
      key=lambda s: int(s.split()[0])
      if s.split()[0].isdigit()
      else float('inf'),
  )


def _prepare_data_for_plotting(data: pd.DataFrame) -> pd.DataFrame:
  """Prepares data for plotting by renaming and reformatting columns."""
  data_plot = data.copy().reset_index(drop=True)
  data_plot['context'] = (
      data_plot['context'].astype(str).map(CONTEXT_LENGTH_TO_LABEL)
  )
  data_plot['steps_ahead'] = data_plot['steps_ahead'].apply(_format_steps_ahead)
  return data_plot


def _get_scales(
    data: pd.DataFrame,
    bins: Sequence[float],
    color_map: dict[str, str] | None = None,
    shape_map: dict[str, str] | None = None,
    exclude: Sequence[str] = (),
) -> tuple[alt.Scale, alt.Scale, alt.Scale, alt.Scale]:
  """Gets the scales for the x, y, color, and shape axes."""
  assert len(bins) > 1

  methods, colors = get_ordered_elements_and_values(
      elements=data['method'].unique(),
      ordered_mapping=METHOD_TO_HEX_COLOR if color_map is None else color_map,
      default_value='#000000',
      excluded_elements=exclude,
  )
  contexts, shapes = get_ordered_elements_and_values(
      elements=data['context'].unique(),
      ordered_mapping=(
          CONTEXT_LABEL_TO_SHAPE if shape_map is None else shape_map),
      default_value='circle',
  )

  x_scale = alt.Scale(**{
      'domain': methods,
  })
  y_scale = alt.Scale(**{
      'bins': bins,
      'clamp': True,
      'domain': (bins[0], bins[-1]),
      'nice': False,
      'padding': 0,
  })
  color_scale = alt.Color(
      'method:N',
      legend=None,
      scale=alt.Scale(domain=methods, range=colors),
      sort=[],
  )
  shape_scale = alt.Shape(
      'context:N',
      legend=None,
      scale=alt.Scale(domain=contexts, range=shapes),
      sort=[],
  )
  return x_scale, y_scale, color_scale, shape_scale


def _get_tooltip_href_kwargs(data: pd.DataFrame, metric: str) -> dict[str, Any]:
  """Gets the kwargs for the tooltip and href."""
  tooltip_href_kwargs = {}
  tooltip_href_kwargs['tooltip'] = ['method', metric]
  if 'url' in data.columns:
    tooltip_href_kwargs['href'] = 'url'
  return tooltip_href_kwargs


def _get_default_scale_bins(metric: str) -> Sequence[float]:
  """Gets the default scale bins for the y-axis."""
  if metric == 'MAE':
    return (0.015, 0.02, 0.025, 0.03, 0.035)
  elif metric == 'MSE':
    return (0.0001, 0.0010, 0.0018, 0.0027, 0.0035)
  else:
    raise ValueError(f'Unsupported metric: {metric}')


def plot_points_and_naive_baselines(
    data: Any,
    metric: str = 'MAE',
    color_map: dict[str, str] | None = None,
    shape_map: dict[str, str] | None = None,
    facet_width: int = 150,
    facet_height: int = 150,
    conditions_as_rows: bool = False,
    bins: Sequence[float] | None = None,
) -> FacetChart:
  """Plots points and naive baselines."""
  _validate_data_for_plotting(
      data, ('context', 'method', 'steps_ahead', 'xid', metric)
  )

  data_plot = _prepare_data_for_plotting(data)
  if bins is None:
    bins = _get_default_scale_bins(metric)

  x_scale, y_scale, color_scale, shape_scale = _get_scales(
      data_plot,
      bins=bins,
      color_map=color_map,
      shape_map=shape_map,
      exclude=('mean', 'stimulus'),
  )
  tooltip_href_kwargs = _get_tooltip_href_kwargs(data_plot, metric)

  points = (
      alt.Chart()
      .mark_point(size=65, opacity=0.95, filled=False)
      .encode(
          x=alt.X('method:N', title='', scale=x_scale),
          y=alt.Y(
              metric,
              scale=y_scale,
          ),
          color=color_scale,
          shape=shape_scale,
          **tooltip_href_kwargs,
      )
      .transform_filter(
          alt.datum.method != 'mean' and alt.datum.method != 'stimulus'
      )
  )
  mean_rule = (
      alt.Chart()
      .mark_rule(opacity=1.0, strokeDash=[3, 3], size=1.5)
      .encode(y=alt.Y(metric), **tooltip_href_kwargs)
      .transform_filter(alt.datum.method == 'mean')
  )
  stimulus_rule = (
      alt.Chart()
      .mark_rule(opacity=1.0, size=1.5)
      .encode(y=alt.Y(metric), **tooltip_href_kwargs)
      .transform_filter(alt.datum.method == 'stimulus')
  )

  return create_facet_chart(
      data_plot,
      (points, mean_rule, stimulus_rule),
      column='steps_ahead',
      column_sort=_sort_steps_ahead(data_plot['steps_ahead'].unique()),
      row='condition' if conditions_as_rows else 'context',
      row_sort=(
          constants.CONDITION_NAMES
          if conditions_as_rows
          else ['short context', 'long context']
      ),
      facet_width=facet_width,
      facet_height=facet_height,
  )


def plot_bars_and_naive_baselines(
    data: Any,
    metric: str = 'MAE',
    color_map: dict[str, str] | None = None,
    shape_map: dict[str, str] | None = None,
    facet_width: int = 150,
    facet_height: int = 150,
    conditions_as_rows: bool = False,
    bins: Sequence[float] | None = None,
) -> FacetChart:
  """Plots bars and naive baselines."""
  _validate_data_for_plotting(
      data, ('context', 'method', 'steps_ahead', 'xid', metric)
  )
  data_plot = _prepare_data_for_plotting(data)

  data_plot_avg_over_conditions = (
      data_plot.groupby(['context', 'method', 'steps_ahead', 'xid'])[metric]
      .mean()
      .to_frame()
      .reset_index()
  )
  if bins is None:
    bins = _get_default_scale_bins(metric)

  x_scale, y_scale, color_scale, _ = _get_scales(
      data_plot,
      bins=bins,
      color_map=color_map,
      shape_map=shape_map,
      exclude=('mean', 'stimulus'),
  )

  num_bars = len(
      [m for m in data['method'].unique() if m not in ('stimulus', 'mean')]
  )
  bars = (
      alt.Chart()
      .mark_bar(opacity=0.4, size=facet_width / num_bars)
      .encode(
          x=alt.X('method:N', title='', scale=x_scale),
          y=alt.Y(
              metric,
              scale=y_scale,
              aggregate='mean',
          ),
          color=color_scale,
      )
      .transform_filter(
          alt.datum.method != 'mean' and alt.datum.method != 'stimulus'
      )
  )
  strokes = (
      alt.Chart()
      .mark_point(
          shape='stroke', size=(facet_width**2 / num_bars**2), opacity=0.95
      )
      .encode(
          x=alt.X('method:N', title='', scale=x_scale),
          y=alt.Y(
              metric,
              scale=y_scale,
              aggregate='mean',
          ),
          color=color_scale,
      )
      .transform_filter(
          alt.datum.method != 'mean' and alt.datum.method != 'stimulus'
      )
  )
  error_bars = (
      alt.Chart()
      .mark_errorbar(extent='ci', opacity=1.0)
      .encode(
          x=alt.X('method:N', title=''),
          y=alt.Y(metric),
          strokeWidth=alt.value(3),
          color=color_scale,
      )
  )
  mean_rule = (
      alt.Chart()
      .mark_rule(opacity=1.0, strokeDash=[3, 3], size=1.5)
      .encode(y=alt.Y(metric, scale=y_scale))
      .transform_filter(alt.datum.method == 'mean')
  )
  stimulus_rule = (
      alt.Chart()
      .mark_rule(opacity=1.0, size=1.5)
      .encode(y=alt.Y(metric, scale=y_scale))
      .transform_filter(alt.datum.method == 'stimulus')
  )
  return create_facet_chart(
      data=data_plot_avg_over_conditions,
      charts=(mean_rule, stimulus_rule, bars, strokes, error_bars),
      column='steps_ahead',
      column_sort=_sort_steps_ahead(data_plot['steps_ahead'].unique()),
      row='condition' if conditions_as_rows else 'context',
      row_sort=(
          constants.CONDITION_NAMES
          if conditions_as_rows
          else ['short context', 'long context']
      ),
      facet_width=facet_width,
      facet_height=facet_height,
  )
