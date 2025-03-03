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

import * as TweakpaneRotationInputPlugin from '@0b5vr/tweakpane-plugin-rotation';
import * as TextareaPlugin from '@pangenerator/tweakpane-textarea-plugin';
import * as EssentialsPlugin from '@tweakpane/plugin-essentials';
import * as THREE from 'three';
import {Pane} from 'tweakpane';

import fragment from '../shader/fragment.glsl?raw'
import vertex from '../shader/vertex.glsl?raw'

import {geometryFunctions} from './geometry.js';
import {PARAMS, STATE} from './globals.js';
import {ZarrLoader} from './loader.js';
import {materialFunctions, prependVertexShaderWithHelpers} from './material.js';
import {CombinedProgress} from './progress.js'
import * as utils from './utils.js';


//
// State
//

function exportState() {
  const url = new URL(window.location.href);
  url.searchParams.set(
      'p', encodeURIComponent(JSON.stringify(pane.exportState())));
  window.history.replaceState(null, null, url);
}

function importState() {
  const url = new URL(window.location.href);
  if (url.searchParams.get('p') != undefined) {
    let decoded = JSON.parse(decodeURIComponent(url.searchParams.get('p')));
    pane.importState(decoded);
  }
}


//
// Tweakpane
//

let pane = new Pane({
  container: document.getElementById('pane'),
  expanded: false,
  title: 'settings',
});

pane.registerPlugin(EssentialsPlugin);
pane.registerPlugin(TextareaPlugin);
pane.registerPlugin(TweakpaneRotationInputPlugin);

const tab = pane.addTab({
  pages: [
    {title: 'controls'},
    {title: 'shader'},
    {title: 'data'},
  ],
});

// controls page
tab.pages[0]
    .addBinding(PARAMS, 'camera', {
      x: {min: -500.0, max: 500.0, step: 1.0},
      y: {min: -500.0, max: 500.0, step: 1.0},
      z: {min: 0.0, max: 2000.0, step: 1.0},
    })
    .on('change', ({value}) => {
      for (var i = 0; i < STATE.views.length; ++i) {
        STATE.views[i].camera.position.x = value.x;
        STATE.views[i].camera.position.y = value.y;
        STATE.views[i].camera.position.z = value.z;
      }
    });
tab.pages[0]
    .addBinding(PARAMS, 'euler', {
      view: 'rotation',
      rotationMode: 'euler',
      order: 'XYZ',
      unit: 'rad',
    })
    .on('change', ({value}) => {
      for (var i = 0; i < STATE.views.length; ++i) {
        STATE.views[i].mesh.rotation.x = value.x;
        STATE.views[i].mesh.rotation.y = value.y;
        STATE.views[i].mesh.rotation.z = value.z;
      }
    });
tab.pages[0].addBinding(PARAMS, 'rotation_delta', {
  dx: {min: -0.1, max: 0.1, step: 0.0025},
  dy: {min: -0.1, max: 0.1, step: 0.0025},
  dz: {min: -0.1, max: 0.1, step: 0.0025},
});
tab.pages[0].addBinding(
    PARAMS, 'frames_per_second', {min: 1, max: 30, step: 1});
tab.pages[0].addBinding(PARAMS, 'stride', {min: 1, max: 10, step: 1});
tab.pages[0].addBinding(PARAMS, 'playback');
tab.pages[0].addBinding(PARAMS, 'loop_forward_backward');

// shader page
tab.pages[1]
    .addBinding(PARAMS, 'background', {view: 'color'})
    .on('change', (ev) => {
      STATE.scene.background = new THREE.Color(PARAMS.background);
    });
tab.pages[1]
    .addBinding(PARAMS, 'vertex', {
      view: 'textarea',
      rows: 10,
      placeholder: vertex,
    })
    .on('change', (ev) => {
      for (var i = 0; i < STATE.views.length; ++i) {
        STATE.views[i].material.vertexShader =
            prependVertexShaderWithHelpers(ev.value);
        STATE.views[i].material.needsUpdate = true;
      }
    });
tab.pages[1]
    .addBinding(PARAMS, 'fragment', {
      view: 'textarea',
      rows: 10,
      placeholder: fragment,
    })
    .on('change', (ev) => {
      for (var i = 0; i < STATE.views.length; ++i) {
        STATE.views[i].material.fragmentShader = ev.value;
        STATE.views[i].material.needsUpdate = true;
      }
    });
// const fpsGraph = tab.pages[1].addBlade({view: 'fpsgraph', label:
// 'fpsgraph',});

// data page
// NOTE: Manually reload page for changes to take effect.
tab.pages[2]
    .addBinding(PARAMS, 'info', {
      view: 'textarea',
      rows: 2,
      placeholder: '',
    })
    .on('change', (ev) => {
      return;
    });
tab.pages[2]
    .addBinding(PARAMS, 'views', {
      view: 'textarea',
      rows: 20,
      placeholder: '',
    })
    .on('change', (ev) => {
      return;
    });
tab.pages[2]
    .addBinding(PARAMS, 'json_src', {
      view: 'textarea',
      rows: 10,
      placeholder: '',
    })
    .on('change', (ev) => {
      return;
    });
tab.pages[2]
    .addBinding(PARAMS, 'zarr_src', {
      view: 'textarea',
      rows: 10,
      placeholder: '',
    })
    .on('change', (ev) => {
      return;
    });
const btn = tab.pages[2].addButton({
  title: 'reload',
});
btn.on('click', () => {
  exportState();
  setTimeout(() => {
    document.location.reload();
  }, 250);
});

utils.addDebugBindings(
    {'STATE': STATE, 'PARAMS': PARAMS, 'pane': pane, 'export': exportState});


//
// Init
//

STATE.scene = utils.initScene();
STATE.scene.background = new THREE.Color(PARAMS.background);

importState();

let params_views = JSON.parse(PARAMS.views);
for (var i = 0; i < params_views.length; ++i) {
  let view = params_views[i];

  var camera = new THREE.PerspectiveCamera(
      50, window.innerWidth / window.innerHeight, 0.01, 10000.);
  camera.position.x = PARAMS.camera.x;
  camera.position.y = PARAMS.camera.y;
  camera.position.z = PARAMS.camera.z;
  camera.layers.set(i + 1);

  STATE.views.push(view);
  STATE.views[i].camera = camera;
  STATE.views[i].texture_idx = 0;
  STATE.views[i].last_increment = 0;
  STATE.views[i].increment_multiplier = 1;
}
STATE.json_src = JSON.parse(PARAMS.json_src);
STATE.zarr_src = JSON.parse(PARAMS.zarr_src);

STATE.renderer = utils.initRenderer(STATE.scene);

for (var i = 0; i < params_views.length; ++i) {
  STATE.views[i].orbit_controls =
      utils.initOrbitControls(STATE.views[i].camera, STATE.renderer);
}


//
// Render
//

function render() {
  for (var i = 0; i < STATE.views.length; ++i) {
    let view = STATE.views[i];

    var left = Math.floor(windowWidth * view.split.left);
    var bottom = Math.floor(windowHeight * view.split.bottom);
    var width = Math.floor(windowWidth * view.split.width);
    var height = Math.floor(windowHeight * view.split.height);
    STATE.renderer.setViewport(left, bottom, width, height);
    STATE.renderer.setScissor(left, bottom, width, height);
    STATE.renderer.setScissorTest(true);

    view.camera.aspect = width / height;
    view.camera.updateProjectionMatrix();

    STATE.renderer.render(STATE.scene, view.camera);
  }
}


//
// Window
//

var windowWidth, windowHeight;

function updateSize() {
  if (windowWidth != window.innerWidth || windowHeight != window.innerHeight) {
    windowWidth = window.innerWidth;
    windowHeight = window.innerHeight;
    STATE.renderer.setSize(windowWidth, windowHeight);
  }
}

updateSize();


//
// Animation
//

let lastExport = 0;

function animate() {
  // fpsGraph.begin();

  const time = performance.now();  // time in milliseconds

  for (var i = 0; i < STATE.views.length; ++i) {
    var view = STATE.views[i];

    if (PARAMS.playback &&
        (time - view.last_increment) >= (1000.0 / PARAMS.frames_per_second)) {
      view.texture_idx += view.increment_multiplier * PARAMS.stride;
      view.last_increment = time;

      if (!PARAMS.loop_forward_backward) {
        if (view.texture_idx > view.textures.length - 1) {
          view.texture_idx = 0;
        }
      } else {
        if (view.texture_idx > view.textures.length - 1) {
          view.increment_multiplier = -1.;
          view.texture_idx = view.textures.length - 1 - PARAMS.stride;
        }
        if (view.texture_idx < 0) {
          view.increment_multiplier = +1.;
          view.texture_idx = PARAMS.stride;
        }
      }
    }

    view.material.uniforms['texture'].value = view.textures[view.texture_idx];
    // view.material.uniforms['time'].value = time * speed % 1.; // Unused.

    view.mesh.rotation.x +=
        view.increment_multiplier * (PARAMS.rotation_delta.x / 100.);
    view.mesh.rotation.y +=
        view.increment_multiplier * (PARAMS.rotation_delta.y / 100.);
    view.mesh.rotation.z +=
        view.increment_multiplier * (PARAMS.rotation_delta.z / 100.);
  }

  render();

  // if ((time - lastExport) >= 1) {
  //   exportState();
  //   lastExport = time;
  // }

  // fpsGraph.end();

  requestAnimationFrame(animate);
}


//
// Launch
//

const manager = new THREE.LoadingManager();
const file_loader = new THREE.FileLoader(manager);

function maybeLaunchScene() {
  if (Object.keys(STATE.json_obj).length !=
          Object.keys(STATE.json_src).length ||
      Object.keys(STATE.zarr_obj).length !=
          Object.keys(STATE.zarr_src).length) {
    return;
  }

  for (var i = 0; i < STATE.views.length; ++i) {
    var view = STATE.views[i];

    var geometry =
        geometryFunctions[view.geometry_fn](...view.geometry_fn_args);
    var material =
        materialFunctions[view.material_fn](...view.material_fn_args);
    view.material = material;

    let mesh = new THREE.Mesh(geometry, material);

    view.mesh = new THREE.Object3D();
    view.mesh.rotation.x = PARAMS.euler.x;
    view.mesh.rotation.y = PARAMS.euler.y;
    view.mesh.rotation.z = PARAMS.euler.z;
    view.mesh.add(mesh);
    view.mesh.layers.set(i + 1);
    view.mesh.traverse(function(child) {
      child.layers.set(i + 1)
    });

    STATE.scene.add(view.mesh);
  }

  utils.addInfoDiv(PARAMS.info, 0.93, 0.0, 1.);

  document.addEventListener('keydown', function(event) {
    if (event.ctrlKey && event.key === 'h') {
      utils.toggleDiv('pane');
    }
  });

  animate();

  document.getElementById('loader').style.display = 'none';
  document.getElementById('pane').style.display = 'initial';
}

const combinedProgress = new CombinedProgress();

combinedProgress.onUpdate(combined => {
  document.getElementById('loaderProgress').textContent =
      `${(0.95 * combined).toFixed(2)}% loaded`;
});

for (const [key, value] of Object.entries(STATE.json_src)) {
  file_loader.load(value, function(data) {
    STATE.json_obj[key] = JSON.parse(data);
    maybeLaunchScene();
  });
}

const zarr_loader = new ZarrLoader(manager);
for (const [key, value] of Object.entries(STATE.zarr_src)) {
  zarr_loader.load(value.store, value.path, function(ds) {
    STATE.zarr_obj[key] = ds;
    maybeLaunchScene();
  }, value.minTime, value.maxTime, value.stepsAhead - 1, combinedProgress);
}
