#include "eri.h"
#include "eri_kernel.h"
#include "hemi/hemi.h"
#include "hemi/parallel_for.h"
#include <iostream>

HEMI_DEV_CALLABLE size_t num_unique_ij(size_t n) { return n * (n + 1) / 2; }
HEMI_DEV_CALLABLE size_t num_unique_ijkl(size_t n) {
  return num_unique_ij(num_unique_ij(n));
}
HEMI_DEV_CALLABLE void triu_ij_from_index(size_t n, size_t index, size_t *i,
                                          size_t *j) {
  size_t a = 1;
  size_t b = (2 * n + 1);
  size_t c = 2 * index;
  size_t i_ = (size_t)((b - std::sqrt(b * b - 4 * a * c)) / (2 * a));
  size_t j_ = size_t(index - (2 * n + 1 - i_) * i_ / 2 + i_);
  *i = i_;
  *j = j_;
}

template <typename FLOAT>
void hartree(const size_t N, const size_t *n, const FLOAT *r, const FLOAT *z,
             const size_t *min_a, const size_t *min_c, const size_t *max_ab,
             const size_t *max_cd, const size_t *Ms, FLOAT *output) {
  std::cout << num_unique_ijkl(N) << std::endl;
  hemi::parallel_for(0, num_unique_ijkl(N), [=] HEMI_LAMBDA(int index) {
    size_t i, j, k, l, ij, kl;
    triu_ij_from_index(num_unique_ij(N), index, &ij, &kl);
    triu_ij_from_index(N, ij, &i, &j);
    triu_ij_from_index(N, kl, &k, &l);
    float out = eri<float>(n[0 * N + i], n[1 * N + i], n[2 * N + i], // a
                           n[0 * N + j], n[1 * N + j], n[2 * N + j], // b
                           n[0 * N + k], n[1 * N + k], n[2 * N + k], // c
                           n[0 * N + l], n[1 * N + l], n[2 * N + l], // d
                           r[0 * N + i], r[1 * N + i], r[2 * N + i], // a
                           r[0 * N + j], r[1 * N + j], r[2 * N + j], // b
                           r[0 * N + k], r[1 * N + k], r[2 * N + k], // c
                           r[0 * N + l], r[1 * N + l], r[2 * N + l], // d
                           z[i], z[j], z[k], z[l],                   // z
                           min_a, min_c, max_ab, max_cd, Ms);
  });
}

template void hartree<float>(const size_t N, const size_t *n, const float *r,
                             const float *z, const size_t *min_a,
                             const size_t *min_c, const size_t *max_ab,
                             const size_t *max_cd, const size_t *Ms,
                             float *output);
