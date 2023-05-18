#ifndef D4FT_NATIVE_GAMMA_IGAMMA_H_
#define D4FT_NATIVE_GAMMA_IGAMMA_H_

#include "constants.h"
#include "digamma.h"
#include "lgamma.h"
#include <cmath>

enum kIgammaMode { VALUE, DERIVATIVE, SAMPLE_DERIVATIVE };

// Helper function for computing Igamma using a power series.
template <typename FLOAT, kIgammaMode mode>
FLOAT IgammaSeries(FLOAT ax, FLOAT x, FLOAT a, bool enabled) {
  // vals: (enabled, r, c, ans, x)
  // 'enabled' is a predication mask that says for which elements we should
  // execute the loop body. Disabled elements have no effect in the loop body.
  // TODO(phawkins): in general this isn't an optimal implementation on any
  // backend. For example, on GPU, we should probably vectorize to the warp
  // size, and then run independent loops for each warp's worth of
  // data.
  //
  FLOAT ans = 1;
  FLOAT r = a;
  FLOAT c = 1;
  FLOAT dc_da = 0;
  FLOAT dans_da = 0;

  while (enabled) {
    r = r + 1;
    dc_da = dc_da * (x / r) + (-1 * c * x) / (r * r);
    dans_da = dans_da + dc_da;
    c = c * (x / r);
    ans = ans + c;
    bool conditional;
    if (mode == VALUE) {
      enabled = enabled && c / ans > eps<FLOAT>;
    } else {
      enabled = enabled && std::abs(dc_da / dans_da) > eps<FLOAT>;
    }
  }
  if (mode == VALUE) {
    return (ans * ax) / a;
  }
  FLOAT dlogax_da = std::log(x) - Digamma(a + 1);
  switch (mode) {
  case DERIVATIVE:
    return ax * (ans * dlogax_da + dans_da) / a;
  case SAMPLE_DERIVATIVE:
  default:
    return -(dans_da + ans * dlogax_da) * x / a;
  }
}

// Helper function for computing Igammac using a continued fraction.
template <typename FLOAT, kIgammaMode mode>
FLOAT IgammacContinuedFraction(FLOAT ax, FLOAT x, FLOAT a, bool enabled) {
  FLOAT y = 1 - a;
  FLOAT z = x + y + 1;
  int c = 0;
  FLOAT pkm2 = 1;
  FLOAT qkm2 = x;
  FLOAT pkm1 = x + 1;
  FLOAT qkm1 = z * x;
  FLOAT ans = pkm1 / qkm1;
  FLOAT t = 1;
  FLOAT dpkm2_da = 0;
  FLOAT dqkm2_da = 0;
  FLOAT dpkm1_da = 0;
  FLOAT dqkm1_da = -x;
  FLOAT dans_da = (dpkm1_da - ans * dqkm1_da) / qkm1;
  while (c < 2000 && enabled) {
    c = c + 1;
    y = y + 1;
    z = z + 2;
    FLOAT yc = y * c;
    FLOAT pk = pkm1 * z - pkm2 * yc;
    FLOAT qk = qkm1 * z - qkm2 * yc;
    FLOAT qk_is_nonzero = qk != 0;
    FLOAT r = pk / qk;
    FLOAT dpk_da = dpkm1_da * z - pkm1 - dpkm2_da * yc + pkm2 * c;
    FLOAT dqk_da = dqkm1_da * z - qkm1 - dqkm2_da * yc + qkm2 * c;
    FLOAT dans_da_new, grad_conditional;
    if (qk_is_nonzero) {
      t = std::abs((ans - r) / r);
      ans = r;
      dans_da_new = (dpk_da - ans * dqk_da) / qk;
      grad_conditional = std::abs(dans_da_new - dans_da);
    } else {
      t = 1;
      ans = ans;
      dans_da_new = dans_da;
      grad_conditional = 1;
    }
    pkm2 = pkm1;
    pkm1 = pk;
    qkm2 = qkm1;
    qkm1 = qk;
    dpkm2_da = dpkm1_da;
    dqkm2_da = dqkm1_da;
    dpkm1_da = dpk_da;
    dqkm1_da = dqk_da;
    bool rescale = std::abs(pk) > 1. / eps<FLOAT>;
    if (rescale) {
      pkm2 = pkm2 * eps<FLOAT>;
      pkm1 = pkm1 * eps<FLOAT>;
      qkm2 = qkm2 * eps<FLOAT>;
      qkm1 = qkm1 * eps<FLOAT>;
      dpkm2_da = dpkm2_da * eps<FLOAT>;
      dpkm1_da = dpkm1_da * eps<FLOAT>;
      dqkm2_da = dqkm2_da * eps<FLOAT>;
      dqkm1_da = dqkm1_da * eps<FLOAT>;
    }
    FLOAT conditional;
    if (mode == VALUE) {
      enabled = (enabled && t > eps<FLOAT>);
    } else {
      enabled = (enabled && grad_conditional > eps<FLOAT>);
    }
  }
  if (mode == VALUE) {
    return ans * ax;
  }
  FLOAT dlogax_da = std::log(x) - Digamma(a);
  switch (mode) {
  case DERIVATIVE:
    return ax * (ans * dlogax_da + dans_da);
  case SAMPLE_DERIVATIVE:
  default:
    return -(dans_da + ans * dlogax_da) * x;
  }
}

template <typename FLOAT> FLOAT Igamma(FLOAT a, FLOAT x) {
  bool is_nan = std::isnan(a) || std::isnan(x);
  FLOAT x_is_zero = x == 0;
  FLOAT x_is_infinity = x == std::numeric_limits<float>::infinity();
  bool domain_error = x < 0 || a <= 0;
  bool use_igammac = (x > 1) && (x > a);
  FLOAT ax = a * std::log(x) - x - Lgamma(a);
  bool underflow = ax < -std::log(std::numeric_limits<FLOAT>::max());
  ax = std::exp(ax);
  bool enabled = !(x_is_zero || domain_error || underflow || is_nan);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  FLOAT output;
  if (use_igammac) {
    output = 1 - IgammacContinuedFraction<FLOAT, VALUE>(ax, x, a,
                                                        enabled && use_igammac);
  } else {
    output = IgammaSeries<FLOAT, VALUE>(ax, x, a, enabled && !use_igammac);
  }
  if (x_is_zero) {
    output = 0;
  }
  if (x_is_infinity) {
    output = 1;
  }
  if (domain_error || is_nan) {
    output = nan;
  }
  return output;
}

template <typename FLOAT> FLOAT IgammaGradA(FLOAT a, FLOAT x) {
  bool is_nan = std::isnan(a) || std::isnan(x);
  bool x_is_zero = x == 0;
  bool domain_error = x < 0 || a <= 0;
  bool use_igammac = (x > 1) && (x > a);
  FLOAT ax = a * std::log(x) - x - Lgamma(a);
  bool underflow = ax < -std::log(std::numeric_limits<FLOAT>::max());
  ax = std::exp(ax);
  bool enabled = !(x_is_zero || domain_error || underflow || is_nan);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  FLOAT output =
      use_igammac ? -IgammacContinuedFraction<DERIVATIVE>(
                        ax, x, a, enabled && use_igammac)
                  : IgammaSeries<DERIVATIVE>(ax, x, a, enabled && !use_igammac);
  output = x_is_zero ? 0 : output;
  output = domain_error || is_nan ? nan : output;
  return output;
}

template <typename FLOAT> FLOAT Igammac(const FLOAT &a, const FLOAT &x) {
  FLOAT max_finite_value = std::numeric_limits<FLOAT>::max();
  bool b;
  bool out_of_range = (x <= 0) || (a <= 0);
  bool use_igamma = (x < 1) || (x < a);
  FLOAT ax = a * std::log(x) - x - std::lgamma(a);
  bool underflow = (ax < -std::log(max_finite_value));
  bool enabled = !(out_of_range || underflow);
  ax = std::exp(ax);
  FLOAT result =
      use_igamma
          ? (1 - IgammaSeries<FLOAT, VALUE>(ax, x, a, enabled && use_igamma))
          : IgammacContinuedFraction<FLOAT, VALUE>(ax, x, a,
                                                   enabled && !use_igamma);
  bool x_is_infinity = (x == std::numeric_limits<FLOAT>::infinity());
  result = x_is_infinity ? 0 : result;
  return out_of_range ? 1 : result;
}

#endif // D4FT_NATIVE_GAMMA_IGAMMA_H_
