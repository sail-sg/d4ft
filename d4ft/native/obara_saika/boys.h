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

#ifndef D4FT_NATIVE_OBARA_SAIKA_BOYS_H_
#define D4FT_NATIVE_OBARA_SAIKA_BOYS_H_

#include "d4ft/native/gamma/igamma.h"
#include "d4ft/native/gamma/lgamma.h"
#include "hemi/hemi.h"
#include <cmath>
#include <limits>

template <typename FLOAT> HEMI_DEV_CALLABLE FLOAT BoysIgamma(FLOAT m, FLOAT T) {
  if (T < std::numeric_limits<FLOAT>::epsilon()) {
    return 1. / (2. * m + 1.);
  } else {
    return 1. / 2. * std::pow(T, (-m - 1. / 2.)) *
           std::exp(Lgamma<FLOAT>(m + 1. / 2.)) * Igamma<FLOAT>(m + 1. / 2., T);
  }
}

#endif // D4FT_NATIVE_OBARA_SAIKA_BOYS_H_
