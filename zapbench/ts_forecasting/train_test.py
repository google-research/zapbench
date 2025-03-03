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

"""Tests for training and evaluation."""

import os
from typing import Any, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from connectomics.common import file
import ml_collections as mlc
import numpy as np
from zapbench.ts_forecasting import train
from zapbench.ts_forecasting.configs import linear
from zapbench.ts_forecasting.configs import mean
from zapbench.ts_forecasting.configs import stimulus
from zapbench.ts_forecasting.configs import tide
from zapbench.ts_forecasting.configs import timemix
from zapbench.ts_forecasting.configs import tsmixer


def build_placeholder_spec(shape: Sequence[int]) -> dict[str, Any]:
  """Builds a placeholder spec containing all ones."""
  return {
      'driver': 'array',
      'dtype': 'float32',
      'array': np.ones(shape).tolist(),
  }


def adjust_config_for_test(
    c: mlc.ConfigDict,
    num_timesteps: int = 1000,
    num_features: int = 8,
    num_static_covariates: int = 2,
    num_dynamic_covariates: int = 2,
    use_static_covariates: bool = False,
    use_past_covariates: bool = False,
    use_future_covariates: bool = False,
    future_covariates_padding: bool = False,
) -> mlc.ConfigDict:
  """Adjusts config for testing."""
  c.train_specs = [{
      'covariates': build_placeholder_spec(
          (num_timesteps, num_dynamic_covariates)
      ),
      'timeseries': build_placeholder_spec((num_timesteps, num_features)),
  }]
  c.val_specs = c.train_specs
  c.infer_spec = c.train_specs[0]

  c.series_spec_shape = (num_timesteps, num_features)
  c.series_shape = (1, c.timesteps_input, num_features)

  covariates = []
  covariates_shapes = []

  if use_static_covariates:
    c.static_covariates_spec = build_placeholder_spec(
        (num_features, num_static_covariates)
    )
    covariates.append('covariates_static')
    covariates_shapes.append((num_features, num_static_covariates))

  if use_past_covariates:
    covariates.append('covariates_input')
    covariates_shapes.append((1, c.timesteps_input, num_dynamic_covariates))

  if use_future_covariates:
    covariates.append('covariates_output')
    covariates_shapes.append((
        1,
        c.timesteps_input if future_covariates_padding else c.timesteps_output,
        num_dynamic_covariates,
    ))

  c.covariates = tuple(covariates)
  c.covariates_shapes = tuple(covariates_shapes)

  c.num_train_steps = 1
  c.num_val_steps = 1
  c.checkpoint_every_steps = 1

  c.infer_sets = [{
      'name': 'test',
      'start_idx': 0,
      'num_windows': 32,
  }]

  c.infer_prefix = 'file://{workdir}'

  return c


class TrainTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='linear',
          config=linear.get_config(),
      ),
      dict(
          testcase_name='mean',
          config=mean.get_config(),
      ),
      dict(
          testcase_name='stimulus',
          config=stimulus.get_config(),
          use_future_covariates=True,
          num_dynamic_covariates=8,
      ),
      dict(
          testcase_name='tide',
          config=tide.get_config(),
          use_static_covariates=True,
          use_past_covariates=True,
          use_future_covariates=True,
      ),
      dict(
          testcase_name='timemix',
          config=timemix.get_config(),
      ),
      dict(
          testcase_name='tsmixer',
          config=tsmixer.get_config(),
      ),
  )

  def test_train_checkpoint_exists(
      self,
      config: mlc.ConfigDict,
      use_static_covariates: bool = False,
      use_past_covariates: bool = False,
      use_future_covariates: bool = False,
      future_covariates_padding: bool = False,
      num_dynamic_covariates: int = 2,
  ):
    self.config = adjust_config_for_test(
        config,
        use_static_covariates=use_static_covariates,
        use_past_covariates=use_past_covariates,
        use_future_covariates=use_future_covariates,
        future_covariates_padding=future_covariates_padding,
        num_dynamic_covariates=num_dynamic_covariates,
    )
    self.tmpdir = self.create_tempdir().full_path

    train.train_and_evaluate(self.config, self.tmpdir)

    checkpoint_dir = os.path.join(self.tmpdir, 'checkpoints')
    self.assertTrue(
        file.Path(checkpoint_dir).exists(),
        'Checkpoint directory not found: %s' % checkpoint_dir,
    )


if __name__ == '__main__':
  absltest.main()
