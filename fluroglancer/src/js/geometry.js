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

import {PARAMS, STATE} from './globals.js';


export const geometryFunctions =
    {
      fromSegmentation: fromSegmentation,
    }

function geometryFromSegmentation(position) {
  let baseGeometry, geometry;

  baseGeometry = new THREE.PlaneGeometry(4, 4, 1, 1);
  geometry = new THREE.InstancedBufferGeometry();
  geometry.index = baseGeometry.index;
  geometry.attributes = baseGeometry.attributes;

  const particleCount = Object.keys(position[0]).length;
  const translateArray = new Float32Array(particleCount * 3);
  const propArray = new Float32Array(particleCount * 2);
  for (let i = 0, i2 = 0, i3 = 0, l = particleCount; i < l;
       i++, i2 += 2, i3 += 3) {
    translateArray[i3 + 0] = position[0][i];
    translateArray[i3 + 1] = position[1][i];
    translateArray[i3 + 2] = position[2][i];
    propArray[i2 + 0] = i;
    propArray[i2 + 1] = 0;  // TODO(jan-matthis): Unused, remove?
  }
  geometry.setAttribute(
      'translate', new THREE.InstancedBufferAttribute(translateArray, 3));
  geometry.setAttribute(
      'prop', new THREE.InstancedBufferAttribute(propArray, 2));
  return geometry;
}

function fromSegmentation(name) {
  return geometryFromSegmentation(
      [
        STATE.json_obj[name].centroid_x, STATE.json_obj[name].centroid_y,
        STATE.json_obj[name].centroid_z
      ],
  );
}
