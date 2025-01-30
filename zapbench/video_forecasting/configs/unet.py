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

"""Model variants for video to video prediction.

Different preprocessed tensors and downsamplings are preconfigured.
"""

from zapbench.video_forecasting import config


def get_config(
    inscale_outscale: str,
) -> config.ml_collections.ConfigDict:
  """Config for vid2vid models with different preprocessed data and scale.

  Args:
    inscale_outscale: comma-separated string for input scale, and output scale.
      Input and output scale are integers indicating the downsampling scale of
      spatial dimensions x and y. Scale of 1 means no downsampling from (2048,
      1328). Scale of 2 means downsampling by 2 to (1024, 664), scale of 3 means
      downsampling by 4 to (512, 332).

  Returns:
    ml_collections.ConfigDict c
  """
  c = config.get_video_prediction_config()

  gcs_driver = 'gcs'
  inscale, outscale = inscale_outscale.split(',')

  inscale = int(inscale) - 1
  outscale = int(outscale) - 1
  ds = 2**outscale
  base_path = 'volumes/20240930/df_over_f_xyz_chunked/s'
  inkvstore = {
      'driver': gcs_driver,
      'bucket': 'zapbench-release',
      'path': base_path + str(inscale),
  }
  outkvstore = {
      'driver': gcs_driver,
      'bucket': 'zapbench-release',
      'path': base_path + str(outscale),
  }
  c.data_config.tensorstore_input_config = {
      'driver': 'zarr3',
      'kvstore': inkvstore,
  }
  c.data_config.tensorstore_output_config = {
      'driver': 'zarr3',
      'kvstore': outkvstore,
  }
  c.data_config.tensorstore_stimulus_config = {
      'driver': 'zarr',
      'kvstore': {
          'driver': gcs_driver,
          'bucket': 'zapbench-release',
          'path': 'volumes/20240930/stimuli_features/'
      },
  }
  c.seg_downsampling = [ds, ds, 1]
  # z needs to be downsampled from 288 -> 72
  c.mask_downsampling = [ds, ds, 4]

  # data config
  c.data_config.x_crop_size = 2048 // ds
  c.data_config.y_crop_size = 1152 // ds
  c.data_config.z_crop_size = 72
  c.data_config.dynamic_range = 1.75  # [-0.25, 1.5]
  return c
