/*
Copyright 2024 The Google Research Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

import fragment from '../shader/fragment.glsl?raw'
import vertex from '../shader/vertex.glsl?raw'

export const BUCKET = 'https://storage.googleapis.com/zapbench-release/';

export const PARAMS = {
  camera: {x: 0.0, y: 0.0, z: 1000.0},
  euler: {x: 0.0, y: 0.0, z: 0.0},
  rotation_delta: {x: 0.0, y: 0.1, z: 0.0},
  frames_per_second: 10.0,
  stride: 1,
  playback: true,
  loop_forward_backward: false,
  background: 0x000000,
  vertex: vertex,
  fragment: fragment,
  json_src: JSON.stringify(
      {'seg': BUCKET + 'fluroglancer/assets/240924_dataframe_centroids.json'}),
  zarr_src: JSON.stringify({
    's1': {
      'store': BUCKET,
      'path': 'volumes/20240930/traces_fluroglancer/',
      minTime: 0,
      maxTime: 649,
    },
  }),
  views: JSON.stringify([
    {
      split: {
        left: 0,
        bottom: 0,
        width: 1.0,
        height: 1.0,
      },
      geometry_fn: 'fromSegmentation',
      geometry_fn_args: ['seg'],
      material_fn: 'fromSeries',
      material_fn_args: [0, 's1'],
    },
  ]),
  info: 'gain condition',
};

export const STATE = {
  views: [],
  json_src: {},
  zarr_src: {},
  json_obj: {},
  zarr_obj: {},
};
