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

"""Hyperparameter utilities."""

import itertools


def fixed(name, value, length=None):
  """Iterator always returning the same value.

  Args:
    name: Parameter name.
    value: Parameter value.
    length: Optional length of iterator.

  Yields:
    A dictionary containing the parameter name and value.
  """
  if length is not None:
    for _ in range(length):
      yield {name: value}
  else:
    while True:
      yield {name: value}


def merge_dicts(*dicts):
  """Merges multiple dictionaries into a single dictionary.

  Args:
    *dicts: An arbitrary number of dictionaries to merge.

  Returns:
    A new dictionary containing all key-value pairs from the input dictionaries.
    If there are duplicate keys, the value from the later dictionary is used.
  """
  merged = {}
  for d in dicts:
    merged.update(d)
  return merged


def product(iterables):
  """Generates the Cartesian product of input iterables as merged dictionaries.

  Args:
    iterables: An arbitrary number of iterable objects.

  Yields:
    A dictionary containing the merged key-value pairs from each combination
    of elements in the input iterables.
  """
  for item in itertools.product(*iterables):
    yield merge_dicts(*item)


def sweep(name, values):
  """Generates a sequence of dictionaries with a single key-value pair.

  Args:
    name: The key to use in each dictionary.
    values: An iterable of values to associate with the key.

  Yields:
    A dictionary with the given key and the next value from the values iterable.
  """
  for value in values:
    yield {name: value}


def zipit(iterables):
  """Generates an iterator as a composition of iterables.

  Args:
    iterables: An arbitrary number of iterable objects.

  Yields:
    A dictionary containing the combined parameter iterator values.
  """
  for item in zip(*iterables):
    yield merge_dicts(*item)
