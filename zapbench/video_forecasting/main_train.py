# Copyright 2025 The Google Research Authors.
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

"""Main file for running video forecasting model training."""

from typing import Sequence

from absl import app
from absl import flags
from connectomics.jax import training
import jax
import ml_collections.config_flags  # pylint:disable=unused-import
from zapbench.video_forecasting import config  # pylint:disable=unused-import
from zapbench.video_forecasting import train


FLAGS = flags.FLAGS

training.define_training_flags()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  training.prep_training()
  train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(main)
