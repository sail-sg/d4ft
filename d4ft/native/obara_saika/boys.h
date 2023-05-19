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
