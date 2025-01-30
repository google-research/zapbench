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

"""Stimulus baseline."""

import dataclasses

from connectomics.jax import config_util
import immutabledict
import ml_collections as mlc
from zapbench import constants
from zapbench import data_utils
from zapbench.models import naive
from zapbench.ts_forecasting.configs import common

_ARGS = immutabledict.immutabledict({
    'covariate_series': '240930_stimulus_evoked_response',
    'seed': -1,  # Model is parameter-free and deterministic.
    'timesteps_input': 1,  # Only used by the data loader, not the model.
})


def get_config(arg: str | None = None) -> mlc.ConfigDict:
  """Default config.

  Args:
    arg: An optional string argument that can be specified on the command line.
      For example `--config=default.py:seed=1` will pass a string containing
      `'seed=1'` to this function which then gets parsed.
      See `_ARGS` for exposed options.

  Returns:
    A `ConfigDict` instance with the complete configuration.
  """
  config = mlc.ConfigDict()

  config.arg = config_util.parse_arg(arg, **_ARGS)

  config.update(common.get_config(**config.arg))

  # No training is required for the stimulus baseline. Run a single step only,
  # after which a checkpoint is saved.
  config.num_train_steps = 1

  config.model_class = 'naive.ReturnCovariates'
  config.return_covariates_config = mlc.ConfigDict(
      dataclasses.asdict(naive.ReturnCovariatesConfig())
  )
  config.covariates = ('covariates_output',)
  config.covariates_shapes = (
      (
          1,
          constants.PREDICTION_WINDOW_LENGTH,
          data_utils.get_covariate_spec(config.covariate_series).shape[-1],
      ),
  )

  return config
