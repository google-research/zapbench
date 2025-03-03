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

"""Config for inference."""

import ml_collections


def get_config(store: str) -> ml_collections.ConfigDict:
  """Returns config for inference starting from training checkpoints.

  Args:
    store: path used for storing inference results

  Returns:
    configuration for infer.py
  """
  c = ml_collections.ConfigDict()
  c.xm_id = 0  # training xm id to load checkpoint from
  c.work_unit = 1  # for example within a grid (defaults to 1 = single exp.)
  c.cpoint_id = -1  # checkpoint id (>0) or otherwise use latest
  c.exp_workdir = ''  # defaults to inference workdir if empty
  c.split = 'test'  # data split in ['train', 'val', 'test', 'test_holdout']
  c.condition = 0  # fish 2.0 condition to evaluate on
  c.write_video_forecast = False
  c.write_video_frequency = 1
  c.write_trace_forecast = False
  c.write_trace_frequency = 1
  c.mask_video = False
  c.write_queue = 50
  c.base_path = store
  c.tensorstore_config = {
      'driver': 'zarr3',
      'kvstore': {'driver': 'file'}
  }
  c.mesh_shape = (1, 1, 1, 1)  # for the manuscript, (1, 1, 4, 4) was used
  c.mesh_names = ('batch', 't', 'x', 'y')
  # how batch dimensions map to mesh axes
  c.mesh_names_batch = ('batch', 't', 'x', 'y')
  c.mesh_names_stimulus = ('batch', 't')
  c.infer_save_json = True
  # {workdir}, {base_path}, {xm_id}, {work_unit}, {cpoint_id} will be replaced
  # if present in `json_path_prefix`.
  c.json_path_prefix = 'file://{base_path}/{xm_id}/{work_unit}/{cpoint_id}'
  return c
