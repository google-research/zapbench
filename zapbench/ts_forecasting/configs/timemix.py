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

"""Time-mix models."""

from collections import abc
import dataclasses

from connectomics.jax import config_util
import immutabledict
import ml_collections as mlc
from zapbench import constants
from zapbench import hparam_utils as hyper
from zapbench.models import tsmixer
from zapbench.ts_forecasting.configs import common


_ARGS = immutabledict.immutabledict({
    'dropout_prob': 0.0,
    'instance_norm': False,
    'n_block': 5,
    'seed': -1,
    'timesteps_input': 4,
})

_EXPERIMENTS = {
    'short_context': hyper.product([
        hyper.sweep('instance_norm', [False,]),
        hyper.sweep('n_block', [5,]),
        hyper.sweep('timesteps_input', [4,]),
        hyper.sweep('seed', [-1, -1, -1]),
    ]),
    'long_context': hyper.product([
        hyper.sweep('instance_norm', [True,]),
        hyper.sweep('n_block', [5,]),
        hyper.sweep('timesteps_input', [256,]),
        hyper.sweep('seed', [-1, -1, -1]),
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

  config.model_class = 'tsmixer.Tsmixer'
  config.tsmixer_config = mlc.ConfigDict(dataclasses.asdict(
      tsmixer.TsmixerConfig(
          activation='relu',
          dropout=config.arg.dropout_prob,
          instance_norm=config.arg.instance_norm,
          feature_mix_residual=True,
          mlp_dim=1,  # Unused
          n_block=config.arg.n_block,
          norm='',
          revert_instance_norm=True,
          time_mix_mlp_dim=-1,
          time_mix_only=True,
          time_mix_residual=True,
          block_residual=False,
          pred_len=constants.PREDICTION_WINDOW_LENGTH)))

  return config


def sweep(add: abc.Callable[..., None], arg: str | None = None) -> None:
  # NOTE: Sweeping multiple configs is not supported in open-source version.
  common.sweep_from_hparams_dict(_EXPERIMENTS, add, arg)
