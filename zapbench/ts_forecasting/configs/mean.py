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

"""Mean baseline."""

from collections import abc
import dataclasses

from connectomics.jax import config_util
import immutabledict
import ml_collections as mlc
from zapbench import constants
from zapbench import hparam_utils as hyper
from zapbench.models import naive
from zapbench.ts_forecasting.configs import common


_ARGS = immutabledict.immutabledict({
    'seed': -1,  # Model is parameter-free and deterministic.
    'timesteps_input': 4,  # Window to average over.
})


_EXPERIMENTS = {
    'context_lengths': hyper.product([
        hyper.sweep('timesteps_input', [1, 2, 4, 8, 16, 32, 64, 128, 256]),
        hyper.sweep('seed', [-1]),
    ]),
}


def get_config(arg: str | None = None) -> mlc.ConfigDict:
  """Default config.

  Args:
    arg: An optional string argument that can be specified on the command line.
      For example `--config=default.py:timesteps_input=4` will pass a string
      containing `'timesteps_input=4'` to this function which then gets parsed.
      See `_ARGS` for exposed options.

  Returns:
    A `ConfigDict` instance with the complete configuration.
  """
  config = mlc.ConfigDict()

  config.arg = config_util.parse_arg(arg, **_ARGS)

  config.update(common.get_config(**config.arg))

  # No training is required for the mean baseline. Run a single step only, after
  # which a checkpoint is saved.
  config.num_train_steps = 1

  config.model_class = 'naive.MeanBaseline'
  config.mean_baseline_config = mlc.ConfigDict(dataclasses.asdict(
      naive.MeanBaselineConfig(pred_len=constants.PREDICTION_WINDOW_LENGTH)))

  return config


def sweep(add: abc.Callable[..., None], arg: str | None = None) -> None:
  # NOTE: Sweeping multiple configs is not supported in open-source version.
  common.sweep_from_hparams_dict(_EXPERIMENTS, add, arg)
