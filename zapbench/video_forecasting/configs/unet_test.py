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

"""Config of heavily downsized setting for local development and testing."""

from zapbench.video_forecasting import config


def get_config() -> config.ml_collections.ConfigDict:
  """Returns a config for local development and testing."""
  c = config.get_video_prediction_config()

  c.log_loss_every_steps = 5
  c.num_eval_steps = 5
  c.eval_every_steps = 5
  c.num_train_steps = 10
  c.train_mask = 'brain'
  c.mesh_shape = (1, 1, 1, 1)

  c.data_config.tensorstore_input_config = {
      'driver': 'zarr3',
      'kvstore': {
          'driver': 'gcs',
          'bucket': 'zapbench-release',
          'path': 'volumes/20240930/df_over_f_xyz_chunked/s2'
      },
  }
  c.data_config.tensorstore_output_config = (
      c.data_config.tensorstore_input_config
  )
  c.data_config.tensorstore_stimulus_config = {
      'driver': 'zarr',
      'kvstore': {
          'driver': 'gcs',
          'bucket': 'zapbench-release',
          'path': 'volumes/20240930/stimuli_features/'
      },
  }
  ds = 4
  c.seg_downsampling = [ds, ds, 1]
  # z needs to be downsampled from 288 -> 72
  c.mask_downsampling = [ds, ds, 4]

  # data config
  c.data_config.x_crop_size = 32
  c.data_config.y_crop_size = 16

  # nunet model config
  c.nunet_config.embed_dim = 8
  c.nunet_config.norm = 'layernorm'
  c.nunet_config.num_maxvit_blocks = 0
  return c
