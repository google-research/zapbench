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

"""Main file for running timeseries forecasting model training."""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags

from zapbench.ts_forecasting import train

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True
)
_WORKDIR = flags.DEFINE_string('workdir', None, 'Work unit directory.')
flags.mark_flags_as_required(['config', 'workdir'])
# Flags --jax_backend_target and --jax_xla_backend are available through JAX.


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.jax_backend_target:
    logging.info('Using JAX backend target %s', FLAGS.jax_backend_target)
    jax_xla_backend = (
        'None' if FLAGS.jax_xla_backend is None else FLAGS.jax_xla_backend
    )
    logging.info('Using JAX XLA backend %s', jax_xla_backend)

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX devices: %r', jax.devices())

  train.train_and_evaluate(FLAGS.config, _WORKDIR.value)


if __name__ == '__main__':
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(main)
