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

import * as THREE from 'three';

import colormaps from '../shader/colormaps.glsl?raw'
import decode from '../shader/decode.glsl?raw'

import {PARAMS, STATE} from './globals.js';
import * as utils from './utils.js';

export const TEXTURE_DIM = 288;


export const materialFunctions =
    {
      fromSeries: fromSeries,
      fromSeriesDifference: fromSeriesDifference,
    }


export function rescaler(value) {
  if (utils.isFloat(value)) {
    return utils.rescaleFloat(value, -0.25, 1.5);
  } else {
    return value / 255.;
  }
}


export function prependVertexShaderWithHelpers(vertex) {
  return decode + colormaps + vertex;
}


export function makeRawShaderMaterial(view_idx) {
  var material = new THREE.RawShaderMaterial({
    uniforms: {
      'time': {value: 0.0},
      'texture': {value: STATE.views[view_idx].textures[0]},
      'resolution':
          {value: new THREE.Vector2(window.innerWidth, window.innerHeight)},
    },
    vertexShader: prependVertexShaderWithHelpers(PARAMS.vertex),
    fragmentShader: PARAMS.fragment,
    depthTest: true,
    depthWrite: true,
  });
  material.transparent = true;
  return material;
}


function fromSeries(view_idx, name, scaling) {
  STATE.views[view_idx].textures = utils.createTextures(
      STATE.zarr_obj[name], TEXTURE_DIM, rescaler, undefined, scaling);
  return makeRawShaderMaterial(view_idx);
}


function fromSeriesDifference(view_idx, name1, name2, scaling) {
  STATE.views[view_idx].textures = utils.createTextures(
      STATE.zarr_obj[name1], TEXTURE_DIM, rescaler, STATE.zarr_obj[name2],
      scaling);
  return makeRawShaderMaterial(view_idx);
}
