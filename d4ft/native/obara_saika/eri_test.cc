// Copyright 2023 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "eri_kernel.h"
#include "hemi/array.h"
#include "hemi/device_api.h"
#include "hemi/hemi.h"
#include "hemi/parallel_for.h"
#include <iostream>

void constant_init(size_t *ptr, size_t size, size_t constant) {
  for (size_t i = 0; i < size; ++i) {
    ptr[i] = constant;
  }
}

int main() {
  size_t N = 200;
  hemi::Array<size_t> n(3 * N, true);
  for (size_t i = 0; i < 3 * N; ++i) {
    n.writeOnlyHostPtr()[i] = rand() % 2;
  }
  hemi::Array<float> r(3 * N, true);
  hemi::Array<float> z(N, true);
  for (size_t i = 0; i < 3 * N; ++i) {
    r.writeOnlyHostPtr()[i] =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  for (size_t i = 0; i < N; ++i) {
    z.writeOnlyHostPtr()[i] =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  hemi::Array<size_t> min_(3, true);
  hemi::Array<size_t> max_(3, true);
  hemi::Array<size_t> max_ab(3, true);
  hemi::Array<size_t> max_cd(3, true);
  hemi::Array<size_t> Ms(3, true);
  constant_init(min_.writeOnlyHostPtr(), 3, 0);
  constant_init(max_.writeOnlyHostPtr(), 3, 0);
  constant_init(max_ab.writeOnlyHostPtr(), 3, 0);
  constant_init(max_cd.writeOnlyHostPtr(), 3, 0);

  size_t(&angular)[3][N] = *reinterpret_cast<size_t(*)[3][N]>(n.hostPtr());

  // min a, c
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < N; ++j) {
      min_.hostPtr()[i] = std::min(min_.hostPtr()[i], angular[i][j]);
      max_.hostPtr()[i] = std::max(max_.hostPtr()[i], angular[i][j]);
    }
  }

  // max ab, cd
  for (size_t i = 0; i < 3; ++i) {
    max_ab.hostPtr()[i] = max_.hostPtr()[i] * 2;
    max_cd.hostPtr()[i] = max_.hostPtr()[i] * 2;
  }

  // Ms
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t s = 0;
      for (size_t k = i; k < 3; ++k) {
        s += angular[k][j];
      }
      Ms.hostPtr()[i] = std::max(Ms.hostPtr()[i], s);
    }
    Ms.hostPtr()[i] = Ms.hostPtr()[i] * 4;
  }
  hartree<float>(N, n.readOnlyPtr(), r.readOnlyPtr(), z.readOnlyPtr(),
                 min_.readOnlyPtr(), min_.readOnlyPtr(), max_ab.readOnlyPtr(),
                 max_cd.readOnlyPtr(), Ms.readOnlyPtr(), nullptr);
}
