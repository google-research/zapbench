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

"""Inference configuration."""

from connectomics.jax import config_util
import immutabledict
import ml_collections as mlc


_ARGS = immutabledict.immutabledict({
    'checkpoint_selection': 'best_val_loss',
    'exp_workdir': '',  # Defaults to inference workdir if empty.
    'infer_save_array': False,
    'infer_save_json': True,
    # {workdir}, {step} will be replaced if present in `infer_prefix`.
    'infer_prefix': 'file://{workdir}/inference/step/{step}',
})


def get_config(arg: str | None = None) -> mlc.ConfigDict:
  """Default config.

  Args:
    arg: An optional string argument that can be specified on the command line.
      For example `--config=infer.py:exp_workdir=...` will pass a string
      containing `'exp_workdir=...'` to this function which then gets parsed.
      See `_ARGS` for exposed options.

  Returns:
    A `ConfigDict` instance which can be used to update an experiment config
    for inference.
  """
  config = mlc.ConfigDict()
  config.args = config_util.parse_arg(arg, **_ARGS)
  config.update(config.args)
  return config
