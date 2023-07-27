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

#ifndef PLAS_GPU_ERI_KERNEL_H_
#define PLAS_GPU_ERI_KERNEL_H_

#include <cmath>

#include "hemi/hemi.h"

template <typename FLOAT>
extern void hartree(const size_t N, const size_t* n, const FLOAT* r,
                    const FLOAT* z, const size_t* min_a, const size_t* min_c,
                    const size_t* max_ab, const size_t* max_cd,
                    const size_t* Ms, FLOAT* output);

#endif
