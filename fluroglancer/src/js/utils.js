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

import colormap from 'colormap';
import * as THREE from 'three';
import {PerspectiveCamera} from 'three';
import {FirstPersonControls} from 'three/addons/controls/FirstPersonControls.js';
import {FlyControls} from 'three/addons/controls/FlyControls.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GUI} from 'three/addons/libs/lil-gui.module.min.js';
import Stats from 'three/addons/libs/stats.module.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {GlitchPass} from 'three/addons/postprocessing/GlitchPass.js';
import {OutputPass} from 'three/addons/postprocessing/OutputPass.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {ShaderPass} from 'three/addons/postprocessing/ShaderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';


export function initScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  return scene;
}

export function addOrthographicCamera(scene) {
  const left = window.innerWidth / -2;
  const right = window.innerWidth / 2;
  const top = window.innerHeight / 2;
  const bottom = window.innerHeight / -2;
  const near = 0.0001;
  const far = 500;
  const camera =
      new THREE.OrthographicCamera(left, right, top, bottom, near, far);
  camera.position.set(0, 0, 100);
  scene.add(camera);
  return camera;
}

export function addPerspectiveCamera(scene) {
  const fov = 50;
  const aspect = window.innerWidth / window.innerHeight;
  const near = 1;
  const far = 50000;
  const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
  camera.position.set(0, 0, 800);
  scene.add(camera);
  return camera;
}

export function initRenderer(scene) {
  const renderer = new THREE.WebGLRenderer();
  renderer.antialias = true;

  renderer.alpha = true;
  // renderer.gammaFactor = 2.2;
  // renderer.outputColorSpace = THREE.SRGBColorSpace; // optional with
  // post-processing

  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

  document.body.appendChild(renderer.domElement);

  window.addEventListener('resize', function() {
    console.log('resizing...');
    var width = window.innerWidth;
    var height = window.innerHeight;
    renderer.setSize(width, height);
  });

  return renderer;
}

export function initStats() {
  const stats = new Stats();
  document.body.appendChild(stats.dom);
  return stats;
}

export function initGui() {
  const gui = new GUI();
  gui.open();
  return gui;
}

export function initFlyControls(camera, renderer) {
  const controls = new FlyControls(camera, renderer.domElement);
  controls.movementSpeed = 100000;
  controls.rollSpeed = Math.PI / 24;
  controls.autoForward = false;
  controls.dragToLook = true;
  return controls;
}

export function initFirstPersonControls(camera, renderer) {
  const controls = new FirstPersonControls(camera, renderer.domElement);
  controls.movementSpeed = 8;
  controls.lookSpeed = 0.08;
  return controls;
}

export function initOrbitControls(camera, renderer) {
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.25;
  return controls;
}

export function addGridHelper(scene, size) {
  const divisions = 10;
  const gridHelper = new THREE.GridHelper(size, divisions);
  scene.add(gridHelper);
}

export function addAxesHelper(scene) {
  const axesHelper = new THREE.AxesHelper(5);
  scene.add(axesHelper);
}

export function addDebugBindings(objs) {
  for (var prop in objs) {
    window[prop] = objs[prop];
  }
}

export function encodeState(state) {
  return encodeURIComponent(JSON.stringify(state));
}

export function decodeState(state) {
  return decodeURIComponent(JSON.parse(state));
}

export function toggleDiv(id) {
  var el = document.getElementById(id);
  el.style.display = el.style.display === 'none' ? '' : 'none';
}

export function addInfoDiv(text, top, left, width) {
  var elem = document.createElement('div');
  elem.classList = ['info'];
  elem.innerText = text;
  elem.style.cssText = [
    'top: ', top * 100.0, '%;', 'left: ', left * 100.0, '%;',
    'width: ', width * 100.0, '%;'
  ].join('');
  document.body.appendChild(elem);
}

export function isFloat(num) {
  return typeof num === 'number' && !Number.isInteger(num);
}

export function rescaleFloat(value, lower, upper) {
  value = Math.min(Math.max(value, lower), upper);
  return (value - lower) / (upper - lower);
}

// To encode/decode floats to RGBA we use bitshifting, see:
// https://stackoverflow.com/a/18454838
export function encodeFloatRGBA(v) {
  const enc = [1.0, 255.0, 65025.0, 16581375.0].map(b => b * v);
  for (let i = 0; i < enc.length; i++) {
    enc[i] = enc[i] - Math.floor(enc[i]);
  }
  const yzww = [enc[1], enc[2], enc[3], enc[3]];
  enc[0] -= yzww[0] / 255.0;
  enc[1] -= yzww[1] / 255.0;
  enc[2] -= yzww[2] / 255.0;
  return enc;
}

export function decodeRGBAFloat(v) {
  const enc = [1.0, 255.0, 65025.0, 16581375.0];
  const dec = [1.0 / enc[0], 1.0 / enc[1], 1.0 / enc[2], 1.0 / enc[3]];
  let result = 0.0;
  for (let i = 0; i < v.length; i++) {
    result += (v[i]) * dec[i];
  }
  return result;
}

export function createTextures(series, dim, rescaler, series2, scaling) {
  // TODO(jan-matthis): Add checks and make more modular.
  let value;
  let value_encoded;
  let diff;
  let textures = [];

  const arraySize = dim * dim;

  let seriesRank = series.shape.length;
  let numTimesteps = series.shape[0];
  let numFeatures = series.shape[seriesRank - 1];
  if (scaling == undefined) {
    scaling = 1.0;
  }

  for (let t = 0; t < numTimesteps; t++) {
    let data = new Uint8Array(arraySize * 4);
    for (let i = 0; i < arraySize; i++) {
      const stride = i * 4;
      if (i < numFeatures) {
        value = rescaler(series.data[t][i]);
        if (series2 != undefined) {
          diff = Math.min(
              Math.max(Math.abs(value - rescaler(series2.data[t][i])), 0.), 1.);
          value_encoded = encodeFloatRGBA(diff * scaling);
        } else {
          value_encoded = encodeFloatRGBA(value * scaling);
        }
        data[stride + 0] = value_encoded[0] * 255;
        data[stride + 1] = value_encoded[1] * 255;
        data[stride + 2] = value_encoded[2] * 255;
        data[stride + 3] = value_encoded[3] * 255;
      }
    }
    let texture = new THREE.DataTexture(data, dim, dim, THREE.RGBAFormat);
    texture.needsUpdate = true;
    textures.push(texture);
  }
  return textures;
}
