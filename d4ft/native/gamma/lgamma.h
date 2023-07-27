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

#ifndef D4FT_NATIVE_GAMMA_LGAMMA_H_
#define D4FT_NATIVE_GAMMA_LGAMMA_H_

#include <array>
#include <cmath>

#include "constants.h"
#include "hemi/hemi.h"

// Compute the Lgamma function using Lanczos' approximation from "A Precision
// Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
// series B. Vol. 1:
// lgamma(z + 1) = (log(2) + log(pi)) / 2 + (z + 1/2) * log(t(z)) - t(z) + A(z)
// t(z) = z + kLanczosGamma + 1/2
// A(z) = kBaseLanczosCoeff + sigma(k = 1, n, kLanczosCoefficients[i] / (z + k))
template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT Lgamma(FLOAT input) {
  FLOAT one_half = 0.5;
  FLOAT one = 1;
  FLOAT pi = M_PI;
  FLOAT log_pi = std::log(M_PI);
  FLOAT log_sqrt_two_pi = (std::log(2) + std::log(M_PI)) / 2;
  FLOAT lanczos_gamma_plus_one_half = kLanczosGamma + 0.5;
  FLOAT log_lanczos_gamma_plus_one_half = std::log(kLanczosGamma + 0.5);
  FLOAT base_lanczos_coeff = kBaseLanczosCoeff;

  // If the input is less than 0.5 use Euler's reflection formula:
  // gamma(x) = pi / (sin(pi * x) * gamma(1 - x))
  FLOAT need_to_reflect = input < one_half;
  FLOAT z = -input * need_to_reflect + (input - one) * (one - need_to_reflect);
  FLOAT x = base_lanczos_coeff;
  for (int i = 0, end = std::size(HEMI_CONSTANT(kLanczosCoefficients)); i < end;
       ++i) {
    FLOAT lanczos_coefficient = HEMI_CONSTANT(kLanczosCoefficients)[i];
    FLOAT index = i;
    x = x + lanczos_coefficient / (z + index + one);
  }

  // To improve accuracy on platforms with less-precise log implementations,
  // compute log(lanczos_gamma_plus_one_half) at compile time and use log1p on
  // the device.
  // log(t) = log(kLanczosGamma + 0.5 + z)
  //        = log(kLanczosGamma + 0.5) + log1p(z / (kLanczosGamma + 0.5))
  FLOAT t = lanczos_gamma_plus_one_half + z;
  FLOAT log_t = log_lanczos_gamma_plus_one_half +
                std::log1p(z / lanczos_gamma_plus_one_half);

  // Compute the final result (modulo reflection).  t(z) may be large, and we
  // need to be careful not to overflow to infinity in the first term of
  //
  //   (z + 1/2) * log(t(z)) - t(z).
  //
  // Therefore we compute this as
  //
  //   (z + 1/2 - t(z) / log(t(z))) * log(t(z)).
  //
  FLOAT log_y =
      log_sqrt_two_pi + (z + one_half - t / log_t) * log_t + std::log(x);

  // Compute the reflected value, used when x < 0.5:
  //
  //   lgamma(x) = log(pi) - lgamma(1-x) - log(abs(sin(pi * x))).
  //
  // (The abs is because lgamma is the log of the absolute value of the gamma
  // function.)
  //
  // We have to be careful when computing the final term above. gamma(x) goes
  // to +/-inf at every integer x < 0, and this is controlled by the
  // sin(pi * x) term.  The slope is large, so precision is particularly
  // important.
  //
  // Because abs(sin(pi * x)) has period 1, we can equivalently use
  // abs(sin(pi * frac(x))), where frac(x) is the fractional part of x.  This
  // is more numerically accurate: It doesn't overflow to inf like pi * x can,
  // and if x is an integer, it evaluates to 0 exactly, which is significant
  // because we then take the log of this value, and log(0) is inf.
  //
  // We don't have a frac(x) primitive in XLA and computing it is tricky, but
  // because abs(sin(pi * x)) = abs(sin(pi * abs(x))), it's good enough for
  // our purposes to use abs(frac(x)) = abs(x) - floor(abs(x)).
  //
  // Furthermore, pi * abs(frac(x)) loses precision when abs(frac(x)) is close
  // to 1.  To remedy this, we can use the fact that sin(pi * x) in the domain
  // [0, 1] is symmetric across the line Y=0.5.
  //
  FLOAT abs_input = std::abs(input);
  FLOAT abs_frac_input = abs_input - std::floor(abs_input);
  // Convert values of abs_frac_input > 0.5 to (1 - frac_input) to improve
  // precision of pi * abs_frac_input for values of abs_frac_input close to 1.
  // FLOAT abs_frac_input_gt_half = abs_frac_input > 0.5;
  // FLOAT reduced_frac_input = abs_frac_input_gt_half * (one - abs_frac_input)
  // +
  //                            (one - abs_frac_input_gt_half) * abs_frac_input;
  FLOAT reduced_frac_input = std::min(abs_frac_input, one - abs_frac_input);
  FLOAT reflection_denom = std::log(std::sin(pi * reduced_frac_input));

  // Avoid computing -inf - inf, which is nan.  If reflection_denom is +/-inf,
  // then it "wins" and the result is +/-inf.
  FLOAT reflection, result;
  if (std::isfinite(reflection_denom)) {
    reflection = (log_pi - reflection_denom - log_y);
  } else {
    reflection = -reflection_denom;
  }
  if (need_to_reflect) {
    result = reflection;
  } else {
    result = log_y;
  }

  // lgamma(+/-inf) = +inf.
  if (std::isinf(input)) {
    return std::numeric_limits<FLOAT>::infinity();
  } else {
    return result;
  }
}

#endif  // D4FT_NATIVE_GAMMA_LGAMMA_H_
