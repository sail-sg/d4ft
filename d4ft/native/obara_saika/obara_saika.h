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

#include <cuda_runtime_api.h>

class Eri4CS8 {
 public:
  static auto ShapeInference(const Spec<int>& angular,
                             const Spec<float>& center,
                             const Spec<float>& exponent) {
    // We put num_gto as the last dimension, as in cuda we parallelize over the
    // num_gto. This layout is more efficient.
    // angular.shape = (3, num_gto)
    // center.shape = (3, num_gto)
    // exponent.shape = (num_gto)
    int num_gto = angular.shape[1];
    // Compute the number of unique 4c integrals.
    return Spec<float>({num_gto, num_gto, num_gto, num_gto});
  }
  static void Cpu(Array<const int>& angular, Array<const float>& center,
                  Array<const float>& exponent, Array<float>& out);
  static void Gpu(cudaStream_t stream,
                  Array<const int>& angular, Array<const float>& center,
                  Array<const float>& exponent, Array<float>& out);
};

class Hartree {
  static auto ShapeInference(const Spec<int>& angular,
                             const Spec<float>& center,
                             const Spec<float>& exponent,
                             const Spec<float>& coeff) {
    // We put num_gto as the last dimension, as in cuda we parallelize over the
    // num_gto. This layout is more efficient.
    // angular.shape = (4, 3, num_gto)
    // center.shape = (4, 3, num_gto)
    // exponent.shape = (4, num_gto)
    // coeff.shape = (num_mo, num_gto)
    int num_gto = angular.shape[2];
    // contracts to a scalar
    return Spec<float>({});
  }
  static void Cpu(Array<const int>& angular, Array<const float>& center,
                  Array<const float>& exponent, Array<float>& out);
  static void Gpu(cudaStream_t stream,
                  Array<const int>& angular, Array<const float>& center,
                  Array<const float>& exponent, Array<float>& out);
};

class Exchange {
  static auto ShapeInference(const Spec<int>& angular,
                             const Spec<float>& center,
                             const Spec<float>& exponent,
                             const Spec<float>& coeff) {
    // We put num_gto as the last dimension, as in cuda we parallelize over the
    // num_gto. This layout is more efficient.
    // angular.shape = (4, 3, num_gto)
    // center.shape = (4, 3, num_gto)
    // exponent.shape = (4, num_gto)
    // coeff.shape = (num_mo, num_gto)
    int num_gto = angular.shape[2];
    // contracts to a scalar
    return Spec<float>({});
  }
  static void Cpu(Array<const int>& angular, Array<const float>& center,
                  Array<const float>& exponent, Array<float>& out);
  static void Gpu(cudaStream_t stream,
                  Array<const int>& angular, Array<const float>& center,
                  Array<const float>& exponent, Array<float>& out);
};
