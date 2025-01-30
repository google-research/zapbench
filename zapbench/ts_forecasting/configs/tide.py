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

"""Tide models."""

from collections import abc
import dataclasses

from connectomics.jax import config_util
import immutabledict
import ml_collections as mlc
from zapbench import constants
from zapbench import data_utils
from zapbench import hparam_utils as hyper
from zapbench.models import tide
from zapbench.ts_forecasting.configs import common


_ARGS = immutabledict.immutabledict({
    'ablate_future_covariates': False,
    'ablate_past_covariates': True,
    'ablate_past_timeseries': False,
    'ablate_static_covariates': True,
    'seed': -1,
    'timesteps_input': 4,
})

_EXPERIMENTS = {
    'short_context': hyper.product([
        hyper.sweep('ablate_past_covariates', [True,]),
        hyper.sweep('ablate_static_covariates', [True,]),
        hyper.sweep('timesteps_input', [4,]),
        hyper.sweep('seed', [-1, -1, -1]),
    ]),
    'long_context': hyper.product([
        hyper.sweep('ablate_past_covariates', [True,]),
        hyper.sweep('ablate_static_covariates', [True,]),
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

  # NOTE: Tide was trained on 16 devices in parallel for the manuscript.
  config.per_device_batch_size = 1

  static_covariates_spec = data_utils.get_position_embedding_spec(
      config.timeseries)
  dynamic_covariates_spec = data_utils.get_covariate_spec(
      config.covariate_series)

  config.covariates = (
      'covariates_static',
      'covariates_input',
      'covariates_output',
  )
  config.covariates_shapes = (
      static_covariates_spec.shape,
      (1, config.timesteps_input, dynamic_covariates_spec.shape[-1]),
      (1, config.timesteps_output, dynamic_covariates_spec.shape[-1]),
  )
  config.static_covariates_spec = static_covariates_spec.to_json()

  config.model_class = 'tide.Tide'
  config.tide_config = mlc.ConfigDict(dataclasses.asdict(
      tide.TideConfig(
          ablate_future_covariates=config.arg.ablate_future_covariates,
          ablate_past_covariates=config.arg.ablate_past_covariates,
          ablate_past_timeseries=config.arg.ablate_past_timeseries,
          ablate_static_covariates=config.arg.ablate_static_covariates,
          activation='relu',
          encoder_decoder_num_hiddens=(128, 128),
          decoder_dim=32,
          dropout_prob=0.0,
          future_covariates_dim=32,
          future_covariates_num_hidden=128,
          instance_norm=False,
          layer_norm=False,
          past_covariates_dim=32,
          past_covariates_num_hidden=128,
          pred_len=constants.PREDICTION_WINDOW_LENGTH,
          temporal_decoder_num_hidden=128,
          use_residual=True)))

  return config


def sweep(add: abc.Callable[..., None], arg: str | None = None) -> None:
  # NOTE: Sweeping multiple configs is not supported in open-source version.
  common.sweep_from_hparams_dict(_EXPERIMENTS, add, arg)
