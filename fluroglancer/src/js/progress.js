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

class CombinedProgress {
  constructor() {
    this.progress = [];
    this.callbacks = [];
  }

  addProgress() {
    this.progress.push(0);
    return this.progress.length - 1;
  }

  updateProgress(index, value) {
    if (index < 0 || index >= this.progress.length) {
      throw new Error('Invalid progress index.');
    }

    this.progress[index] = value;
    this.calculateCombinedProgress();
  }

  calculateCombinedProgress() {
    if (this.progress.length === 0) {
      return 0;
    }

    const sum = this.progress.reduce((a, b) => a + b, 0);
    const combined = sum / this.progress.length;

    this.triggerCallbacks(combined);
  }

  onUpdate(callback) {
    this.callbacks.push(callback);
  }

  triggerCallbacks(combined) {
    this.callbacks.forEach(callback => callback(combined));
  }
}

export {CombinedProgress};
