#include "eri.h"
#include "eri_kernel.h"
#include "hemi/hemi.h"

void constant_init(int *ptr, int size, int constant) {
  for (int i = 0; i < size; ++i) {
    ptr[i] = constant;
  }
}

int num_unique_ij(int n) { return int(n * (n + 1) / 2); }
int num_unique_ijkl(int n) { return num_unique_ij(num_unique_ij(n)); }
void triu_ij_from_index(int n, int index, int *i, int *j) {
  int a = 1;
  int b = -1 * (2 * n + 1);
  int c = 2 * index;
  int i_ = (int)((-b - sqrt(b * b - 4 * a * c)) / (2 * a));
  int j_ = int(index - (2 * n + 1 - i_) * i_ / 2 + i_);
  *i = i_;
  *j = j_;
}

template <typename FLOAT>
void hartree(const int N, const FLOAT *n, const FLOAT *r, const FLOAT *z,
             const int *min_a, const int *min_c, const int *max_ab,
             const int *max_cd, const int *Ms, FLOAT *output) {
  hemi::parallel_for(0, num_unique_ijkl(N), [&] HEMI_LAMBDA(int index) {
    int ij, kl, i, j, k, l;
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
