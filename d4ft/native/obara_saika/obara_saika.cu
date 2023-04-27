#include "obara_saika.h"

void Eri4CS8::Gpu(cudaStream_t stream,
                  Array<const int>& angular, Array<const float>& center,
                  Array<const float>& exponent, Array<float>& out) {
  int num_gto = angular.spec->shape[2];
  auto K = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };
    int idx = cta * nv + tid;
    // TODO(shizk): compute the i, j, k, l from idx
    // the index i needs to be the inner most loop.
    out[idx] = eri(angular.ptr[i + 0*num_gto],
                   angular.ptr[i + 1*num_gto],
                   angular.ptr[i + 2*num_gto],
                   angular.ptr[j + 0*num_gto],
                   angular.ptr[j + 1*num_gto],
                   angular.ptr[j + 2*num_gto],
                   angular.ptr[k + 0*num_gto],
                   angular.ptr[k + 1*num_gto],
                   angular.ptr[k + 2*num_gto],
                   angular.ptr[l + 0*num_gto],
                   angular.ptr[l + 1*num_gto],
                   angular.ptr[l + 2*num_gto],
                   R[idx], R[batch+idx], R[batch*2+idx],
        R[batch*3+idx], R[batch*4+idx], R[batch*5+idx],
        R[batch*6+idx], R[batch*7+idx], R[batch*8+idx],
        R[batch*9+idx], R[batch*10+idx], R[batch*11+idx],
        Z[idx], Z[batch+idx], Z[batch*2+idx], Z[batch*3+idx],
        min_a, min_c, max_ab, max_cd, Ms);
  }
}

void Hartree::Gpu() {
}

void Exchange::Gpu() {
}
