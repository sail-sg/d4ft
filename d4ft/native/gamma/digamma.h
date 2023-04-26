#ifndef D4FT_NATIVE_GAMMA_DIGAMMA_H_
#define D4FT_NATIVE_GAMMA_DIGAMMA_H_

#include <cmath>
#include "constants.h"

template <typename FLOAT>
FLOAT Digamma(FLOAT input) {
  FLOAT zero = 0;
  FLOAT one_half = 0.5;
  FLOAT one = 1;
  FLOAT pi = M_PI;
  FLOAT lanczos_gamma = kLanczosGamma;
  FLOAT lanczos_gamma_plus_one_half = kLanczosGamma + 0.5;
  FLOAT log_lanczos_gamma_plus_one_half = std::log(kLanczosGamma + 0.5);
  FLOAT base_lanczos_coeff = kBaseLanczosCoeff;

  // If the input is less than 0.5 use Euler's reflection formula:
  // digamma(x) = digamma(1 - x) - pi * cot(pi * x)
  bool need_to_reflect = input < one_half;
  FLOAT z;
  if (need_to_reflect) {
    z = -input;
  } else {
    z = input - one;
  }
  FLOAT num = zero;
  FLOAT denom = base_lanczos_coeff;
  for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
    FLOAT lanczos_coefficient = kLanczosCoefficients[i];
    FLOAT index = i;
    num = num - lanczos_coefficient / ((z + index + one) * (z + index + one));
    denom = denom + lanczos_coefficient / (z + index + one);
  }

  // To improve accuracy on platforms with less-precise log implementations,
  // compute log(lanczos_gamma_plus_one_half) at compile time and use log1p on
  // the device.
  // log(t) = log(kLanczosGamma + 0.5 + z)
  //        = log(kLanczosGamma + 0.5) + log1p(z / (kLanczosGamma + 0.5))
  FLOAT t = lanczos_gamma_plus_one_half + z;
  FLOAT log_t = log_lanczos_gamma_plus_one_half +
      std::log1p(z / lanczos_gamma_plus_one_half);

  FLOAT y = log_t + num / denom - lanczos_gamma / t;

  // We need to be careful how we compute cot(pi * input) below: For
  // near-integral values of `input`, pi * input can lose precision.
  //
  // Input is already known to be less than 0.5 (otherwise we don't have to
  // reflect).  We shift values smaller than -0.5 into the range [-.5, .5] to
  // increase precision of pi * input and the resulting cotangent.
  FLOAT reduced_input = input + std::abs(std::floor(input + 0.5));
  FLOAT reflection =
      y - pi * std::cos(pi * reduced_input) / std::sin(pi * reduced_input);
  FLOAT real_result;
  if (need_to_reflect) {
    real_result = reflection;
  } else {
    real_result = y;
  }

  // Digamma has poles at negative integers and zero; return nan for those.
  if (input <= zero && input == std::floor(input)) {
    return std::numeric_limits<FLOAT>::quiet_NaN();
  } else {
    return real_result;
  }
}

#endif // D4FT_NATIVE_GAMMA_DIGAMMA_H_
