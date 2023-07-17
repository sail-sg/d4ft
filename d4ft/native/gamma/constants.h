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

#ifndef D4FT_NATIVE_GAMMA_CONSTANTS_H_
#define D4FT_NATIVE_GAMMA_CONSTANTS_H_

#include "hemi/hemi.h"
#include <array>
#include <limits>

// Coefficients for the Lanczos approximation of the gamma function. The
// coefficients are uniquely determined by the choice of g and n (kLanczosGamma
// and kLanczosCoefficients.size() + 1). The coefficients below correspond to
// [7, 9]. [5, 7], [7, 9], [9, 10], and [607/128.0, 15] were evaluated and [7,
// 9] seemed to be the least sensitive to the quality of the log function. In
// particular, [5, 7] is the only choice where -1.5e-5 <= lgamma(2) <= 1.5e-5
// for a particularly inaccurate log function.
static constexpr double kLanczosGamma = 7; // aka g
static constexpr double kBaseLanczosCoeff = 0.99999999999980993227684700473478;

#define K_LANCZOS_COEFFICIENTS                                                 \
  {                                                                            \
    676.520368121885098567009190444019, -1259.13921672240287047156078755283,   \
        771.3234287776530788486528258894, -176.61502916214059906584551354,     \
        12.507343278686904814458936853, -0.13857109526572011689554707,         \
        9.984369578019570859563e-6, 1.50563273514931155834e-7                  \
  }

typedef double LanczosCoefficientsType[8];
HEMI_DEFINE_STATIC_CONSTANT(LanczosCoefficientsType kLanczosCoefficients,
                            K_LANCZOS_COEFFICIENTS);

#if !defined(HEMI_CUDA_DISABLE) && defined(__CUDACC__)
template <typename FLOAT>
static __constant__ FLOAT eps_devconst = std::numeric_limits<FLOAT>::epsilon();
template <typename FLOAT>
static FLOAT eps_hostconst = std::numeric_limits<FLOAT>::epsilon();
#else
template <typename FLOAT>
static FLOAT eps_hostconst = std::numeric_limits<FLOAT>::epsilon();
#endif

#endif // D4FT_NATIVE_GAMMA_CONSTANTS_H_
