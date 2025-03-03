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

import {Loader} from 'three';
import * as zarr from 'zarr';


class ZarrLoader extends Loader {
  constructor(manager) {
    super(manager);
  }

  load(store, path, onLoad, minTime, maxTime, stepsAhead, combinedProgress) {
    const progressbarIdx = combinedProgress.addProgress();

    const options = {
      concurrencyLimit: 25,
      progressCallback: ({progress, queueSize}) => {
        const currentProgress = progress / queueSize * 100
        combinedProgress.updateProgress(progressbarIdx, currentProgress);
      }
    };

    // TODO(jan-matthis): Support caching and block multiple requests
    zarr.openArray({store: store, path: path, mode: 'r'})
        .then(ds => {
          if (ds.shape.length == 2) {
            // Assume time x feature.
            return ds.get(
                [
                  zarr.slice(
                      (minTime == undefined) ? 0 : minTime,
                      (maxTime == undefined) ? ds.shape[0] : maxTime),
                  zarr.slice(0, ds.shape[1])
                ],
                options);
          } else if (ds.shape.length == 3) {
            // Assume window x steps_ahead x feature.
            return ds.get(
                [
                  zarr.slice(0, ds.shape[0]), stepsAhead,
                  zarr.slice(0, ds.shape[2])
                ],
                options);
          } else {
            console.log('Unsupported shape.');
          }
        })
        .then(
            data => {
              onLoad(data);
            }  // TODO(jan-matthis): Handle onProgress, onError.
        )
  }
}

export {ZarrLoader};
