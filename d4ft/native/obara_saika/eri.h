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

#ifndef D4FT_NATIVE_OBARA_SAIKA_ERI_H_
#define D4FT_NATIVE_OBARA_SAIKA_ERI_H_

#include <cstring>

#include "boys.h"
#include "comb.h"
#include "hemi/hemi.h"

#define MAX_XYZ 4 * 6
#define MAX_YZ 4 * 4
#define MAX_Z 4 * 2
#define MAX_CD 4
#define MAX_AB 4

template <typename FLOAT>
HEMI_DEV_CALLABLE void vertical_0_0_c_0(size_t i, const FLOAT* pq,
                                        const FLOAT* qc, FLOAT I[][MAX_XYZ + 1],
                                        FLOAT rho, FLOAT eta,
                                        const size_t* max_cd,
                                        const size_t* Ms) {
  FLOAT* I_0_cm2 = I[MAX_CD];
  FLOAT* I_0_cm1 = I[0];
  FLOAT wq_i = rho * pq[i] / eta;
  for (size_t j = 0; j < max_cd[i]; ++j) {
    FLOAT cm1 = (FLOAT)j;
    FLOAT* I_0_c = I[j + 1];
    for (size_t k = 0; k <= Ms[i]; ++k) {
      FLOAT I_mp1_k =
          wq_i * I_0_cm1[k] + (-cm1 / 2 / eta * rho / eta) * I_0_cm2[k];
      I_0_c[k] = qc[i] * I_0_cm1[k] + (cm1 / 2 / eta) * I_0_cm2[k];
      if (k > 0) {
        I_0_c[k - 1] += I_mp1_k;
      }
    }
    I_0_cm2 = I_0_cm1;
    I_0_cm1 = I_0_c;
  }
}

template <typename FLOAT>
HEMI_DEV_CALLABLE void vertical_a_0_c_0(
    size_t i, FLOAT I[][MAX_XYZ + 1], const FLOAT* ab, const FLOAT* cd,
    const FLOAT* pa, const FLOAT* pq, FLOAT rho, FLOAT zeta, FLOAT eta,
    const size_t* na, const size_t* nb, const size_t* nc, const size_t* nd,
    const size_t* Ms, const size_t* min_a, const size_t* min_c,
    const size_t* max_ab, const size_t* max_cd, FLOAT* out) {
  FLOAT cache[MAX_CD + 1][MAX_XYZ + 1] = {0};
  FLOAT wa[MAX_AB + 1], wc[MAX_CD + 1];
  for (size_t j = 0; j <= max_ab[i]; ++j) {
    FLOAT mask = (FLOAT)(j >= na[i] && j <= na[i] + nb[i]);
    wa[j] = mask * HEMI_CONSTANT(comb)[nb[i]][j - na[i]] *
            pow(ab[i], nb[i] - j + na[i]);
  }
  for (size_t j = 0; j <= max_cd[i]; ++j) {
    FLOAT mask = (FLOAT)(j >= nc[i] && j <= nc[i] + nd[i]);
    wc[j] = mask * HEMI_CONSTANT(comb)[nd[i]][j - nc[i]] *
            pow(cd[i], nd[i] - j + nc[i]);
  }
  FLOAT(*I_am2)[MAX_XYZ + 1] = cache;
  FLOAT(*I_am1)[MAX_XYZ + 1] = I;
  FLOAT(*I_a)[MAX_XYZ + 1] = I_am2;
  for (size_t j = 0; j <= max_ab[i]; ++j) {
    FLOAT am1 = (FLOAT)j;
    FLOAT wp_i = -rho * pq[i] / zeta;
    for (size_t k = 0; k <= max_cd[i]; ++k) {
      for (size_t l = 0; l <= Ms[i]; ++l) {
        FLOAT I_mp1_kl =
            wp_i * I_am1[k][l] + (-am1 / 2 / zeta * rho / zeta) * I_am2[k][l];
        // inplace write
        I_a[k][l] = pa[i] * I_am1[k][l] + (am1 / 2 / zeta) * I_am2[k][l];
        if (l > 0) {
          I_a[k][l - 1] += I_mp1_kl;
        }
      }
    }
    for (size_t k = 0; k < max_cd[i]; ++k) {
      for (size_t l = 1; l <= Ms[i]; ++l) {
        I_a[k + 1][l - 1] += (k + 1) * I_am1[k][l] / 2 / (zeta + eta);
      }
    }
    if (j >= min_a[i]) {
      FLOAT(*I_j)[MAX_XYZ + 1] = I_am1;
      for (size_t k = min_c[i]; k <= max_cd[i]; ++k) {
        for (size_t l = 0; l <= Ms[i + 1]; ++l) {
          out[l] += wa[j] * I_j[k][l] * wc[k];
        }
      }
    }
    I_am2 = I_am1;
    I_am1 = I_a;
    I_a = I_am2;
  }
}

template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT zeta(FLOAT za, FLOAT zb) {
  return za + zb;
}

template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT xi(FLOAT za, FLOAT zb) {
  return za * zb / zeta(za, zb);
}

template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT rp(FLOAT ra, FLOAT rb, FLOAT za, FLOAT zb) {
  return (za * ra + zb * rb) / (za + zb);
}

template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT rho(FLOAT zeta, FLOAT eta) {
  return zeta * eta / (zeta + eta);
}

template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT T(FLOAT rho, FLOAT* pq) {
  return rho * (pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2]);
}

// translate above python code into C code
template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT K(FLOAT z1, FLOAT z2, FLOAT* r1, FLOAT* r2) {
  FLOAT d_squared = 0;
  for (size_t i = 0; i < 3; ++i) {
    d_squared += (r1[i] - r2[i]) * (r1[i] - r2[i]);
  }
  return std::sqrt((FLOAT)2.) * std::pow(M_PI, (FLOAT)(5. / 4.)) / (z1 + z2) *
         std::exp(-z1 * z2 * d_squared / (z1 + z2));
}

template <typename FLOAT>
HEMI_DEV_CALLABLE FLOAT
eri(size_t nax, size_t nay, size_t naz, size_t nbx, size_t nby, size_t nbz,
    size_t ncx, size_t ncy, size_t ncz, size_t ndx, size_t ndy, size_t ndz,
    FLOAT rax, FLOAT ray, FLOAT raz, FLOAT rbx, FLOAT rby, FLOAT rbz, FLOAT rcx,
    FLOAT rcy, FLOAT rcz, FLOAT rdx, FLOAT rdy, FLOAT rdz, FLOAT za, FLOAT zb,
    FLOAT zc, FLOAT zd, const size_t* min_a, const size_t* min_c,
    const size_t* max_ab, const size_t* max_cd, const size_t* Ms) {
  size_t na[3] = {nax, nay, naz};
  size_t nb[3] = {nbx, nby, nbz};
  size_t nc[3] = {ncx, ncy, ncz};
  size_t nd[3] = {ndx, ndy, ndz};
  FLOAT ra[3] = {rax, ray, raz};
  FLOAT rb[3] = {rbx, rby, rbz};
  FLOAT rc[3] = {rcx, rcy, rcz};
  FLOAT rd[3] = {rdx, rdy, rdz};

  FLOAT rp_[3], rq_[3], pa_[3], pb_[3], qc_[3], qd_[3], ab_[3], cd_[3], pq_[3];
  for (size_t i = 0; i < 3; ++i) {
    rp_[i] = rp(ra[i], rb[i], za, zb);
    rq_[i] = rp(rc[i], rd[i], zc, zd);
    pa_[i] = rp_[i] - ra[i];
    pb_[i] = rp_[i] - rb[i];
    qc_[i] = rq_[i] - rc[i];
    qd_[i] = rq_[i] - rd[i];
    ab_[i] = ra[i] - rb[i];
    cd_[i] = rc[i] - rd[i];
    pq_[i] = rp_[i] - rq_[i];
  }
  FLOAT zeta_ = zeta(za, zb);
  FLOAT eta_ = zeta(zc, zd);
  FLOAT rho_ = rho(zeta_, eta_);
  FLOAT T_ = T(rho_, pq_);
  FLOAT k_ab = K(za, zb, ra, rb);
  FLOAT k_cd = K(zc, zd, rc, rd);
  FLOAT prefactor = std::pow(zeta_ + eta_, (FLOAT)(-1. / 2.)) * k_ab * k_cd;

  FLOAT I[MAX_CD + 1][MAX_XYZ + 1] = {0};
  FLOAT out[MAX_YZ + 1] = {0};
  // set I[0] to Boys
  for (int i = 0; i <= Ms[0]; ++i) {
    I[0][i] = BoysIgamma<FLOAT>(i, T_);
  }
  vertical_0_0_c_0(0, pq_, qc_, I, rho_, eta_, max_cd, Ms);
  vertical_a_0_c_0(0, I, ab_, cd_, pa_, pq_, rho_, zeta_, eta_, na, nb, nc, nd,
                   Ms, min_a, min_c, max_ab, max_cd, out);
  // reset I to zero
  std::memset(I, 0, sizeof(I));
  // set I[0] to out;
  for (size_t i = 0; i <= Ms[1]; ++i) {
    I[0][i] = out[i];
  }
  // set out[:MAX_Z+1] = {0};
  std::memset(out, 0, sizeof(out));
  vertical_0_0_c_0(1, pq_, qc_, I, rho_, eta_, max_cd, Ms);
  vertical_a_0_c_0(1, I, ab_, cd_, pa_, pq_, rho_, zeta_, eta_, na, nb, nc, nd,
                   Ms, min_a, min_c, max_ab, max_cd, out);
  // reset I to zero
  std::memset(I, 0, sizeof(I));
  // set I[0] to out;
  for (size_t i = 0; i <= Ms[2]; ++i) {
    I[0][i] = out[i];
  }
  // set out[:1] = {0};
  std::memset(out, 0, sizeof(out));
  vertical_0_0_c_0(2, pq_, qc_, I, rho_, eta_, max_cd, Ms);
  vertical_a_0_c_0(2, I, ab_, cd_, pa_, pq_, rho_, zeta_, eta_, na, nb, nc, nd,
                   Ms, min_a, min_c, max_ab, max_cd, out);
  return out[0] * prefactor * 0.5;
}

#endif  // D4FT_NATIVE_OBARA_SAIKA_ERI_H_
