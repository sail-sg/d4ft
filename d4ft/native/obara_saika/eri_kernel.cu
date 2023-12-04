#include "eri.h"
#include "eri_kernel.h"
#include "hemi/hemi.h"
#include "hemi/parallel_for.h"
#include <iostream>

HEMI_DEV_CALLABLE int num_unique_ij(int n) { return n * (n + 1) / 2; }
HEMI_DEV_CALLABLE int num_unique_ijkl(int n) {
  return num_unique_ij(num_unique_ij(n));
}
HEMI_DEV_CALLABLE void triu_ij_from_index(int n, int index, int *i,
                                          int *j) {
  int a = 1;
  int b = (2 * n + 1);
  int c = 2 * index;
  int i_ = (int)((b - std::sqrt(b * b - 4 * a * c)) / (2 * a));
  int j_ = int(index - (2 * n + 1 - i_) * i_ / 2 + i_);
  *i = i_;
  *j = j_;
}

// template <typename FLOAT>
void Hartree_32::Gpu(cudaStream_t stream, 
                  Array<const int>& N, 
                  Array<const int>& index_4c, 
                  Array<const int>& n, 
                  Array<const float>& r, 
                  Array<const float>& z,
                  Array<const int>& min_a, 
                  Array<const int>& min_c, 
                  Array<const int>& max_ab,
                  Array<const int>& max_cd, 
                  Array<const int>& Ms, 
                  Array<const int>& ab_range,
                  Array<float>& output) {
  std::cout<<index_4c.spec->shape[0]<<std::endl;
  hemi::ExecutionPolicy ep;
  ep.setStream(stream);
  hemi::parallel_for(ep, 0, index_4c.spec->shape[0], [=] HEMI_LAMBDA(int index) {
    int i, j, k, l, ij, kl;
    // triu_ij_from_index(num_unique_ij(N.ptr[0]), index_4c.ptr[index], &ij, &kl);
    // triu_ij_from_index(N.ptr[0], ij, &i, &j);
    // triu_ij_from_index(N.ptr[0], kl, &k, &l);
    // output.ptr[index] = index_4c.ptr[index];
    i = index_4c.ptr[4*index + 0];
    j = index_4c.ptr[4*index + 1];
    k = index_4c.ptr[4*index + 2];
    l = index_4c.ptr[4*index + 3];
    output.ptr[index] = eri<float>(n.ptr[0 * N.ptr[0] + i], n.ptr[1 * N.ptr[0] + i], n.ptr[2 * N.ptr[0] + i], // a
                           n.ptr[0 * N.ptr[0] + j], n.ptr[1 * N.ptr[0] + j], n.ptr[2 * N.ptr[0] + j], // b
                           n.ptr[0 * N.ptr[0] + k], n.ptr[1 * N.ptr[0] + k], n.ptr[2 * N.ptr[0] + k], // c
                           n.ptr[0 * N.ptr[0] + l], n.ptr[1 * N.ptr[0] + l], n.ptr[2 * N.ptr[0] + l], // d
                           r.ptr[0 * N.ptr[0] + i], r.ptr[1 * N.ptr[0] + i], r.ptr[2 * N.ptr[0] + i], // a
                           r.ptr[0 * N.ptr[0] + j], r.ptr[1 * N.ptr[0] + j], r.ptr[2 * N.ptr[0] + j], // b
                           r.ptr[0 * N.ptr[0] + k], r.ptr[1 * N.ptr[0] + k], r.ptr[2 * N.ptr[0] + k], // c
                           r.ptr[0 * N.ptr[0] + l], r.ptr[1 * N.ptr[0] + l], r.ptr[2 * N.ptr[0] + l], // d
                           z.ptr[i], z.ptr[j], z.ptr[k], z.ptr[l],                   // z
                           min_a.ptr, min_c.ptr, max_ab.ptr, max_cd.ptr, Ms.ptr);
  });
}

void Hartree_64::Gpu(cudaStream_t stream, 
                  Array<const int>& N,
                  Array<const int>& screened_length,  
                  Array<const int>& n, 
                  Array<const double>& r, 
                  Array<const double>& z,
                  Array<const int>& min_a, 
                  Array<const int>& min_c, 
                  Array<const int>& max_ab,
                  Array<const int>& max_cd, 
                  Array<const int>& Ms, 
                  Array<const int>& sorted_ab_idx,
                  Array<const int>& sorted_cd_idx,
                  Array<const int>& screened_cd_idx_start,
                  Array<const int>& screened_idx_offset,
                  Array<int>& output) {
  // Prescreening
  int* idx_4c;
  int idx_length;
  cudaMemcpy(&idx_length, screened_length.ptr, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMalloc((void **)&idx_4c, 2 * idx_length * sizeof(int));
  std::cout<<idx_length<<std::endl;
  int num_cd = sorted_cd_idx.spec->shape[0];

  // Pre-screen, result is (ab_index, cd_index), i.e. (ab, cd)
  hemi::ExecutionPolicy ep;
  ep.setStream(stream);
  hemi::parallel_for(ep, 0, screened_cd_idx_start.spec->shape[0], [=] HEMI_LAMBDA(int index) {
    for(int i = screened_cd_idx_start.ptr[index]; i < num_cd; i++ ){
      int loc;
      loc = screened_idx_offset.ptr[index] + i - screened_cd_idx_start.ptr[index];
      idx_4c[loc] = sorted_ab_idx.ptr[index]; // ab
      idx_4c[loc + screened_length.ptr[0]] = sorted_cd_idx.ptr[i]; // cd
      output.ptr[loc] = sorted_ab_idx.ptr[index]; // ab
      output.ptr[loc + screened_length.ptr[0]] = sorted_cd_idx.ptr[i]; // cd
    }
    __syncthreads();
  });

  // Now we have ab cd, we can compute eri and contract it to output
  // For contract, we need 1. count 2. pgto normalization coeff 3. pgto coeff 4.rdm1 (Mocoeff)
  hemi::parallel_for(ep, 0, idx_length, [=] HEMI_LAMBDA(int index) {
    int a, b, c, d; // pgto 4c idx
    int i, j, k, l; // cgto 4c idx
    double eri_result;
    triu_ij_from_index(N.ptr[0], idx_4c[index], &a, &b);
    triu_ij_from_index(N.ptr[0], idx_4c[index + screened_length.ptr[0]], &c, &d);
    eri_result = eri<double>(n.ptr[0 * N.ptr[0] + a], n.ptr[1 * N.ptr[0] + a], n.ptr[2 * N.ptr[0] + a], // a
                           n.ptr[0 * N.ptr[0] + b], n.ptr[1 * N.ptr[0] + b], n.ptr[2 * N.ptr[0] + b], // b
                           n.ptr[0 * N.ptr[0] + c], n.ptr[1 * N.ptr[0] + c], n.ptr[2 * N.ptr[0] + c], // c
                           n.ptr[0 * N.ptr[0] + d], n.ptr[1 * N.ptr[0] + d], n.ptr[2 * N.ptr[0] + d], // d
                           r.ptr[0 * N.ptr[0] + a], r.ptr[1 * N.ptr[0] + a], r.ptr[2 * N.ptr[0] + a], // a
                           r.ptr[0 * N.ptr[0] + b], r.ptr[1 * N.ptr[0] + b], r.ptr[2 * N.ptr[0] + b], // b
                           r.ptr[0 * N.ptr[0] + c], r.ptr[1 * N.ptr[0] + c], r.ptr[2 * N.ptr[0] + c], // c
                           r.ptr[0 * N.ptr[0] + d], r.ptr[1 * N.ptr[0] + d], r.ptr[2 * N.ptr[0] + d], // d
                           z.ptr[a], z.ptr[b], z.ptr[c], z.ptr[d],                   // z
                           min_a.ptr, min_c.ptr, max_ab.ptr, max_cd.ptr, Ms.ptr);
  });

  // std::cout<<index_4c.spec->shape[0]<<std::endl;
  // hemi::ExecutionPolicy ep;
  // ep.setStream(stream);
  // hemi::parallel_for(ep, 0, index_4c.spec->shape[0], [=] HEMI_LAMBDA(int index) {
  //   int i, j, k, l, ij, kl;
  //   // triu_ij_from_index(num_unique_ij(N.ptr[0]), index_4c.ptr[index], &ij, &kl);
  //   // triu_ij_from_index(N.ptr[0], ij, &i, &j);
  //   // triu_ij_from_index(N.ptr[0], kl, &k, &l);
  //   // output.ptr[index] = index_4c.ptr[index];
  //   i = index_4c.ptr[4*index + 0];
  //   j = index_4c.ptr[4*index + 1];
  //   k = index_4c.ptr[4*index + 2];
  //   l = index_4c.ptr[4*index + 3];
  //   output.ptr[index] = eri<double>(n.ptr[0 * N.ptr[0] + i], n.ptr[1 * N.ptr[0] + i], n.ptr[2 * N.ptr[0] + i], // a
  //                          n.ptr[0 * N.ptr[0] + j], n.ptr[1 * N.ptr[0] + j], n.ptr[2 * N.ptr[0] + j], // b
  //                          n.ptr[0 * N.ptr[0] + k], n.ptr[1 * N.ptr[0] + k], n.ptr[2 * N.ptr[0] + k], // c
  //                          n.ptr[0 * N.ptr[0] + l], n.ptr[1 * N.ptr[0] + l], n.ptr[2 * N.ptr[0] + l], // d
  //                          r.ptr[0 * N.ptr[0] + i], r.ptr[1 * N.ptr[0] + i], r.ptr[2 * N.ptr[0] + i], // a
  //                          r.ptr[0 * N.ptr[0] + j], r.ptr[1 * N.ptr[0] + j], r.ptr[2 * N.ptr[0] + j], // b
  //                          r.ptr[0 * N.ptr[0] + k], r.ptr[1 * N.ptr[0] + k], r.ptr[2 * N.ptr[0] + k], // c
  //                          r.ptr[0 * N.ptr[0] + l], r.ptr[1 * N.ptr[0] + l], r.ptr[2 * N.ptr[0] + l], // d
  //                          z.ptr[i], z.ptr[j], z.ptr[k], z.ptr[l],                   // z
  //                          min_a.ptr, min_c.ptr, max_ab.ptr, max_cd.ptr, Ms.ptr);
  // });
}

// template <typename FLOAT>
void Hartree_32_uncontracted::Gpu(cudaStream_t stream, 
                  Array<const int>& N, 
                  Array<const int>& index_4c, 
                  Array<const int>& n, 
                  Array<const float>& r, 
                  Array<const float>& z,
                  Array<const int>& min_a, 
                  Array<const int>& min_c, 
                  Array<const int>& max_ab,
                  Array<const int>& max_cd, 
                  Array<const int>& Ms, 
                  Array<float>& output) {
  // std::cout<<index_4c.spec->shape[0]<<std::endl;
  hemi::ExecutionPolicy ep;
  ep.setStream(stream);
  hemi::parallel_for(ep, 0, index_4c.spec->shape[0], [=] HEMI_LAMBDA(int index) {
    int i, j, k, l, ij, kl;
    // triu_ij_from_index(num_unique_ij(N.ptr[0]), index_4c.ptr[index], &ij, &kl);
    // triu_ij_from_index(N.ptr[0], ij, &i, &j);
    // triu_ij_from_index(N.ptr[0], kl, &k, &l);
    // output.ptr[index] = index_4c.ptr[index];
    i = index_4c.ptr[4*index + 0];
    j = index_4c.ptr[4*index + 1];
    k = index_4c.ptr[4*index + 2];
    l = index_4c.ptr[4*index + 3];
    output.ptr[index] = eri<float>(n.ptr[0 * N.ptr[0] + i], n.ptr[1 * N.ptr[0] + i], n.ptr[2 * N.ptr[0] + i], // a
                           n.ptr[0 * N.ptr[0] + j], n.ptr[1 * N.ptr[0] + j], n.ptr[2 * N.ptr[0] + j], // b
                           n.ptr[0 * N.ptr[0] + k], n.ptr[1 * N.ptr[0] + k], n.ptr[2 * N.ptr[0] + k], // c
                           n.ptr[0 * N.ptr[0] + l], n.ptr[1 * N.ptr[0] + l], n.ptr[2 * N.ptr[0] + l], // d
                           r.ptr[0 * N.ptr[0] + i], r.ptr[1 * N.ptr[0] + i], r.ptr[2 * N.ptr[0] + i], // a
                           r.ptr[0 * N.ptr[0] + j], r.ptr[1 * N.ptr[0] + j], r.ptr[2 * N.ptr[0] + j], // b
                           r.ptr[0 * N.ptr[0] + k], r.ptr[1 * N.ptr[0] + k], r.ptr[2 * N.ptr[0] + k], // c
                           r.ptr[0 * N.ptr[0] + l], r.ptr[1 * N.ptr[0] + l], r.ptr[2 * N.ptr[0] + l], // d
                           z.ptr[i], z.ptr[j], z.ptr[k], z.ptr[l],                   // z
                           min_a.ptr, min_c.ptr, max_ab.ptr, max_cd.ptr, Ms.ptr);
  });
}

void Hartree_64_uncontracted::Gpu(cudaStream_t stream, 
                  Array<const int>& N,
                  Array<const int>& index_4c,  
                  Array<const int>& n, 
                  Array<const double>& r, 
                  Array<const double>& z,
                  Array<const int>& min_a, 
                  Array<const int>& min_c, 
                  Array<const int>& max_ab,
                  Array<const int>& max_cd, 
                  Array<const int>& Ms, 
                  Array<double>& output) {
  // std::cout<<index_4c.spec->shape[0]<<std::endl;
  hemi::ExecutionPolicy ep;
  ep.setStream(stream);
  hemi::parallel_for(ep, 0, index_4c.spec->shape[0], [=] HEMI_LAMBDA(int index) {
    int i, j, k, l, ij, kl;
    // triu_ij_from_index(num_unique_ij(N.ptr[0]), index_4c.ptr[index], &ij, &kl);
    // triu_ij_from_index(N.ptr[0], ij, &i, &j);
    // triu_ij_from_index(N.ptr[0], kl, &k, &l);
    // output.ptr[index] = index_4c.ptr[index];
    i = index_4c.ptr[4*index + 0];
    j = index_4c.ptr[4*index + 1];
    k = index_4c.ptr[4*index + 2];
    l = index_4c.ptr[4*index + 3];
    output.ptr[index] = eri<double>(n.ptr[0 * N.ptr[0] + i], n.ptr[1 * N.ptr[0] + i], n.ptr[2 * N.ptr[0] + i], // a
                           n.ptr[0 * N.ptr[0] + j], n.ptr[1 * N.ptr[0] + j], n.ptr[2 * N.ptr[0] + j], // b
                           n.ptr[0 * N.ptr[0] + k], n.ptr[1 * N.ptr[0] + k], n.ptr[2 * N.ptr[0] + k], // c
                           n.ptr[0 * N.ptr[0] + l], n.ptr[1 * N.ptr[0] + l], n.ptr[2 * N.ptr[0] + l], // d
                           r.ptr[0 * N.ptr[0] + i], r.ptr[1 * N.ptr[0] + i], r.ptr[2 * N.ptr[0] + i], // a
                           r.ptr[0 * N.ptr[0] + j], r.ptr[1 * N.ptr[0] + j], r.ptr[2 * N.ptr[0] + j], // b
                           r.ptr[0 * N.ptr[0] + k], r.ptr[1 * N.ptr[0] + k], r.ptr[2 * N.ptr[0] + k], // c
                           r.ptr[0 * N.ptr[0] + l], r.ptr[1 * N.ptr[0] + l], r.ptr[2 * N.ptr[0] + l], // d
                           z.ptr[i], z.ptr[j], z.ptr[k], z.ptr[l],                   // z
                           min_a.ptr, min_c.ptr, max_ab.ptr, max_cd.ptr, Ms.ptr);
  });
}

// HEMI_DEV_CALLABLE size_t num_unique_ij(size_t n) { return n * (n + 1) / 2; }
// HEMI_DEV_CALLABLE size_t num_unique_ijkl(size_t n) {
//   return num_unique_ij(num_unique_ij(n));
// }
// HEMI_DEV_CALLABLE void triu_ij_from_index(size_t n, size_t index, size_t *i,
//                                           size_t *j) {
//   size_t a = 1;
//   size_t b = (2 * n + 1);
//   size_t c = 2 * index;
//   size_t i_ = (size_t)((b - std::sqrt(b * b - 4 * a * c)) / (2 * a));
//   size_t j_ = size_t(index - (2 * n + 1 - i_) * i_ / 2 + i_);
//   *i = i_;
//   *j = j_;
// }

// template <typename FLOAT>
// void hartree(const size_t N, const size_t *n, const FLOAT *r, const FLOAT *z,
//              const size_t *min_a, const size_t *min_c, const size_t *max_ab,
//              const size_t *max_cd, const size_t *Ms, FLOAT *output) {
//   std::cout << num_unique_ijkl(N) << std::endl;
//   hemi::parallel_for(0, num_unique_ijkl(N), [=] HEMI_LAMBDA(int index) {
//     size_t i, j, k, l, ij, kl;
//     triu_ij_from_index(num_unique_ij(N), index, &ij, &kl);
//     triu_ij_from_index(N, ij, &i, &j);
//     triu_ij_from_index(N, kl, &k, &l);
//     float out = eri<float>(n[0 * N + i], n[1 * N + i], n[2 * N + i], // a
//                            n[0 * N + j], n[1 * N + j], n[2 * N + j], // b
//                            n[0 * N + k], n[1 * N + k], n[2 * N + k], // c
//                            n[0 * N + l], n[1 * N + l], n[2 * N + l], // d
//                            r[0 * N + i], r[1 * N + i], r[2 * N + i], // a
//                            r[0 * N + j], r[1 * N + j], r[2 * N + j], // b
//                            r[0 * N + k], r[1 * N + k], r[2 * N + k], // c
//                            r[0 * N + l], r[1 * N + l], r[2 * N + l], // d
//                            z[i], z[j], z[k], z[l],                   // z
//                            min_a, min_c, max_ab, max_cd, Ms);
//   });
// }

// template void hartree<float>(const size_t N, const size_t *n, const float *r,
//                              const float *z, const size_t *min_a,
//                              const size_t *min_c, const size_t *max_ab,
//                              const size_t *max_cd, const size_t *Ms,
//                              float *output);
