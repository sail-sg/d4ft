/*
 * Copyright 2023 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPECS_H_
#define SPECS_H_

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

using Shape = std::vector<int>;

/**
 * @brief Calculate the total number of elements in a tensor / array, given **
 * the shape.
 **/
static int Prod(const int* shape, std::size_t ndim) {
  return std::accumulate(shape, shape + ndim, static_cast<int>(1),
                         std::multiplies<>());
}

/**
 * @brief Specifications (data type, shape) of a tensor / array. ** Used on the
 * C++/CUDA side.
 * */
template <typename D>
class Spec {
 public:
  using dtype = D;  // NOLINT

  std::vector<int> shape;

  // constructors
  explicit Spec(std::vector<int>&& shape) : shape(std::move(shape)) {}
  explicit Spec(const std::vector<int>& shape) : shape(shape) {}
  explicit Spec() = default;

  // total number of elements
  int Size() { return Prod(shape.data(), shape.size()); }
};

/**
 * @brief Tensor / Array. Contains the pointer to the actual data, and the Spec.
 * */
template <typename T>
struct Array {
  T* ptr;                       // data
  Spec<std::decay_t<T>>* spec;  // (dtype, shape).
  // NOTE: decay_t is used here to remove type transformations
};

#endif  // SPECS_H_
