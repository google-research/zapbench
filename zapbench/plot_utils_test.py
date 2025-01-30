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

"""Tests for plotting utilities."""

from absl.testing import absltest
import altair as alt
import pandas as pd
from zapbench import plot_utils


class PlotUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.data = pd.DataFrame({
        'context': [4, 4, 256, 256, 4, 256, 4, 256],
        'method': [
            'linear',
            'tide',
            'linear',
            'tide',
            'mean',
            'stimulus',
            'new_method',
            'linear',
        ],
        'steps_ahead': [1, 1, 4, 4, 1, 4, 4, 4],
        'xid': [1, 2, 3, 4, 5, 6, 7, 8],
        'MAE': [0.02, 0.03, 0.025, 0.035, 0.01, 0.02, 0.04, 0.018],
        'MSE': [0.001, 0.002, 0.0015, 0.0025, 0.0005, 0.001, 0.003, 0.0008],
        'condition': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B'],
        'url': ['url://' for _ in range(8)],
    })

  def test_register_and_enable_custom_theme(self):
    plot_utils.register_and_enable_custom_theme()
    self.assertEqual(alt.themes.active, 'custom')

  def test_format_and_sort_steps_ahead(self):
    self.assertEqual(plot_utils._format_steps_ahead(1), '1 step ahead')
    self.assertEqual(plot_utils._format_steps_ahead(3), '3 steps ahead')

    steps = ['3 steps ahead', '1 step ahead', '10 steps ahead', '2 steps ahead']
    sorted_steps = plot_utils._sort_steps_ahead(steps)
    self.assertEqual(
        sorted_steps,
        ['1 step ahead', '2 steps ahead', '3 steps ahead', '10 steps ahead'],
    )

  def test_create_facet_chart(self):
    chart = alt.Chart(self.data).mark_point().encode(x='context', y='MAE')
    facet_chart = plot_utils.create_facet_chart(
        self.data, [chart], column='method', row='steps_ahead'
    )
    self.assertIsInstance(facet_chart, alt.FacetChart)

  def test_plot_points_and_naive_baselines(self):
    chart = plot_utils.plot_points_and_naive_baselines(self.data, 'MAE')
    self.assertIsInstance(chart, alt.FacetChart)

  def test_plot_bars_and_naive_baselines(self):
    chart = plot_utils.plot_bars_and_naive_baselines(self.data, 'MAE')
    self.assertIsInstance(chart, alt.FacetChart)


if __name__ == '__main__':
  absltest.main()
