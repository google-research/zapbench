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

"""Constants for ZAPBench."""
# pylint: disable=line-too-long

# Marks the beginning of condition 0 to 8; final number is the end of timeseries
CONDITION_OFFSETS = (0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879)

# Number of timesteps excluded at the beginning and end of each condition
CONDITION_PADDING = 1

# Condition indices of training, holdout, and both conditions
CONDITIONS_TRAIN = (0, 1, 2, 4, 5, 6, 7, 8)
CONDITIONS_HOLDOUT = (3,)
CONDITIONS = CONDITIONS_TRAIN + CONDITIONS_HOLDOUT

# Condition names
CONDITION_NAMES = ('gain', 'dots', 'flash', 'taxis', 'turning', 'position',
                   'open loop', 'rotation', 'dark')

# Recommended fraction of timesteps per condition used as validation set
VAL_FRACTION = 0.1

# Fraction of timesteps per condition that are used as test set
TEST_FRACTION = 0.2

# Maximum number of input timesteps that any given algorithm can use
MAX_CONTEXT_LENGTH = 256

# Length of the prediction window for testing/evaluation
PREDICTION_WINDOW_LENGTH = 32

# Name of the timeseries used for the benchmark; key in `SPECS`
TIMESERIES_NAME = '240930_traces'

# Dictionary of TensorStore specs containing timeseries forecasting data
SPECS = {
    '240930_traces': {
        'kvstore': 'gs://zapbench-release/volumes/20240930/traces/',
        'driver': 'zarr3',
        'transform': {
            'input_exclusive_max': [[7879], 71721],
            'input_inclusive_min': [0, 0],
            'input_labels': ['t', 'f'],
        }
    },
}

# Minimum and maximum values for timeseries
MIN_MAX_VALUES = {
    '240930_traces': (-0.25, 1.5),
}

# Segmentation dataframes associated with timeseries
SEGMENTATION_DATAFRAMES = {
    '240930_traces': 'gs://zapbench-release/volumes/20240930/segmentation/dataframe.json',
}

# Position embeddings of neurons
POSITION_EMBEDDING_SPECS = {
    '240930_traces': {
        'kvstore': 'gs://zapbench-release/volumes/20240930/position_embedding/',
        'driver': 'zarr',
        'rank': 2,
        'metadata': {'shape': [71721, 192]},
        'transform': {
            'input_inclusive_min': [0, 0],
            'input_exclusive_max': [[71721], [192]],
            'input_labels': ['f', 'a'],
        },
    }
}

# Name of covariate series used for the benchmark; key in `COVARIATE_SPECS`
COVARIATE_SERIES_NAME = '240930_stimuli_features'

# Covariates computed using stimulus information
COVARIATE_SPECS = {
    '240930_stimuli_features': {
        'kvstore': 'gs://zapbench-release/volumes/20240930/stimuli_features/',
        'driver': 'zarr',
        'rank': 2,
        'metadata': {'shape': [7879, 26]},
        'transform': {
            'input_inclusive_min': [0, 0],
            'input_exclusive_max': [[7879], [26]],
            'input_labels': ['t', 'f'],
        },
    },
    '240930_stimulus_evoked_response': {
        'kvstore': 'gs://zapbench-release/volumes/20240930/stimulus_evoked_response/',
        'driver': 'zarr3',
        'transform': {
            'input_exclusive_max': [[7879], 71721],
            'input_inclusive_min': [0, 0],
            'input_labels': ['t', 'f'],
        }
    },
}

# Rastermap sortings of timeseries as JSON files and associated specs
RASTERMAP_SORTINGS = {
    '240930_traces': 'gs://zapbench-release/volumes/20240930/traces_rastermap_sorted/sorting.json',
}
RASTERMAP_SPECS = {
    '240930_traces': {
        'kvstore': 'gs://zapbench-release/volumes/20240930/traces_rastermap_sorted/s0/',
        'driver': 'zarr3',
        'transform': {
            'input_exclusive_max': [[7879], 71721],
            'input_inclusive_min': [0, 0],
            'input_labels': ['t', 'f'],
        }
    }
}
