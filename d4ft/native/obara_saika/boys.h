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

#include <cmath>
#include <limits>

#include "d4ft/native/gamma/igamma.h"
#include "d4ft/native/gamma/lgamma.h"
#include "hemi/hemi.h"
#include "comb.h"

template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT BoysIgamma(FLOAT m, FLOAT T) {
  if (T < std::numeric_limits<FLOAT>::epsilon()) {
    return 1. / (2. * m + 1.);
  } else {
    return 1. / 2. * std::pow(T, (-m - 1. / 2.)) *
           std::exp(Lgamma<FLOAT>(m + 1. / 2.)) * Igamma<FLOAT>(m + 1. / 2., T);
  }
}

template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT neville(FLOAT tresult0, FLOAT tresult1, FLOAT tdt0, FLOAT tdt1) {
        return (-tdt1) * (tresult0 - tresult1) / (tdt0 - tdt1) + tresult1;
}

// Need debug
template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT BoysNeville(FLOAT m, FLOAT T) {
    const int ngrids[4] = {100, 400, 1600, 6400};
    // const int total_size = 8500; 
    FLOAT tdts[4];
    // FLOAT t[total_size];

    // int t_index = 0;
    for (int j = 0; j < 4; ++j) {
        tdts[j] = 1.0 / static_cast<FLOAT>(ngrids[j]);
        // for (int i = 0; i < ngrids[j]; ++i) {
        //     t[t_index++] = static_cast<FLOAT>(i) / static_cast<FLOAT>(ngrids[j]);
        // }
    }

    FLOAT left_endpoint = (static_cast<int>(m) == 0) ? 0.5 : 0.0;
    FLOAT right_endpoint = std::exp(-T) / 2;

    // FLOAT boys_vals[total_size];
    // for (int i = 0; i < total_size; ++i) {
    //     boys_vals[i] = std::exp(-T * t[i] * t[i]) * std::pow(t[i], 2 * m);
    // }

    int idx[5] = {0};
    int sum = 0;
    for (int j = 0; j < 4; ++j) {
        sum += ngrids[j];
        idx[j + 1] = sum;
    }

    FLOAT tresults[4];
    for (int i = 0; i < 4; ++i) {
        FLOAT sum = 0.0;
        for (int j = idx[i]; j < idx[i + 1]; ++j) {
            // sum += boys_vals[j];
            FLOAT t_j = static_cast<FLOAT>(j-idx[i]) / static_cast<FLOAT>(ngrids[i]);
            sum += std::exp(-T * t_j * t_j) * std::pow(t_j, 2 * m);
        }
        tresults[i] = (left_endpoint + sum + right_endpoint) * tdts[i];
    }

    FLOAT tresult01 = neville(tresults[0], tresults[1], tdts[0], tdts[1]);
    FLOAT tresult12 = neville(tresults[1], tresults[2], tdts[1], tdts[2]);
    FLOAT tresult23 = neville(tresults[2], tresults[3], tdts[2], tdts[3]);
    FLOAT tresult012 = neville(tresult01, tresult12, tdts[0], tdts[2]);
    FLOAT tresult123 = neville(tresult12, tresult23, tdts[1], tdts[3]);
    FLOAT result = neville(tresult012, tresult123, tdts[0], tdts[3]);

    return result;
}

template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT BoysPrecomp(FLOAT m, FLOAT T) {
    bool pred = T > 27.0;
    int m_idx = static_cast<int> (m);

    if (!pred) {
        // Process for small T
        int idx0 = static_cast<int>(T * 100);
        FLOAT x0 = idx0 / 100.0;
        FLOAT y0 = boysfuns[m_idx][idx0];
        int idx1 = idx0 + 1;
        FLOAT x1 = x0 + 0.01;
        FLOAT y1 = boysfuns[m_idx][idx1];
        int idx2 = idx0 + 2;
        FLOAT x2 = x0 + 0.02;
        FLOAT y2 = boysfuns[m_idx][idx2];
        int idx3 = idx0 + 3;
        FLOAT x3 = x0 + 0.03;
        FLOAT y3 = boysfuns[m_idx][idx3];
        int idx4 = idx0 + 4;
        FLOAT x4 = x0 + 0.04;
        FLOAT y4 = boysfuns[m_idx][idx4];

        // Neville 5-point interpolation
        FLOAT y01 = (T - x1) * (y0 - y1) / (x0 - x1) + y1;
        FLOAT y12 = (T - x2) * (y1 - y2) / (x1 - x2) + y2;
        FLOAT y23 = (T - x3) * (y2 - y3) / (x2 - x3) + y3;
        FLOAT y34 = (T - x4) * (y3 - y4) / (x3 - x4) + y4;
        FLOAT y012 = (T - x2) * (y01 - y12) / (x0 - x2) + y12;
        FLOAT y123 = (T - x3) * (y12 - y23) / (x1 - x3) + y23;
        FLOAT y234 = (T - x4) * (y23 - y34) / (x2 - x4) + y34;
        FLOAT y0123 = (T - x3) * (y012 - y123) / (x0 - x3) + y123;
        FLOAT y1234 = (T - x4) * (y123 - y234) / (x1 - x4) + y234;
        FLOAT y01234 = (T - x4) * (y0123 - y1234) / (x0 - x4) + y1234;

        return y01234;
    } else {
        // Process for large T
        return boysasympoticconstant[m_idx] * pow(T, -m_idx - 0.5);
    }
}

template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT BoysMacLaurin(FLOAT m, FLOAT T) {
    assert(m < 100);
    int m_idx = static_cast<int> (m);
    FLOAT T_crit = std::numeric_limits<FLOAT>::is_bounded == true ? -log( std::numeric_limits<FLOAT>::min() * 100.5 / 2. ) : FLOAT(0) ;
    // if (std::numeric_limits<FLOAT>::is_bounded && T > T_crit) {
    //   throw std::overflow_error("FmEval_Reference<double>::eval: double lacks precision for the given value of argument T");
    // }
    FLOAT half = FLOAT(1)/2;
    FLOAT denom = (m_idx + half);
    using std::exp;
    FLOAT term = exp(-T) / (2 * denom);
    FLOAT old_term = 0;
    FLOAT sum = term;
    FLOAT epsilon = 1e-16; // get_epsilon(T);
    FLOAT epsilon_divided_10 = epsilon / 10;
    do {
      denom += 1;
      old_term = term;
      term = old_term * T / denom;
      sum += term;
      //rel_error = term / sum , hence iterate until rel_error = epsilon
      // however, must ensure that contributions are decreasing to ensure that omitted contributions are smaller than epsilon
    } while (term > sum * epsilon_divided_10 || old_term < term);

    return sum;
}

#endif  // D4FT_NATIVE_OBARA_SAIKA_BOYS_H_
