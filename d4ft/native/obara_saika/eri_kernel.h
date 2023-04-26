#ifndef PLAS_GPU_ERI_KERNEL_H_
#define PLAS_GPU_ERI_KERNEL_H_

#include "gpu/memory.h"
#include "gpu/loadstore.h"
#include "gpu/kernel_launch.h"
#include "eri.h"

namespace plas {
namespace gpu {

template <typename launch_arg_t = empty_t, typename input_it, typename input_it_2, typename output_it>
void eri_batch(input_it N, input_it_2 R, input_it_2 Z, int batch,
  int* min_a, int* min_c, int* max_ab, int* max_cd, int* Ms,
  output_it output, context_t& context) {
  typedef typename conditional_typedef_t<
      launch_arg_t, launch_params_t<512, 1> >::type_t launch_t;
  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(batch);
  auto K = [=] PLAS_DEVICE(int tid, int cta) { 
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };
    int idx = cta * nv + tid;
    output[idx] = eri(N[idx], N[batch+idx], N[batch*2+idx],
        N[batch*3+idx], N[batch*4+idx], N[batch*5+idx],
        N[batch*6+idx], N[batch*7+idx], N[batch*8+idx],
        N[batch*9+idx], N[batch*10+idx], N[batch*11+idx],
        R[idx], R[batch+idx], R[batch*2+idx],
        R[batch*3+idx], R[batch*4+idx], R[batch*5+idx],
        R[batch*6+idx], R[batch*7+idx], R[batch*8+idx],
        R[batch*9+idx], R[batch*10+idx], R[batch*11+idx],
        Z[idx], Z[batch+idx], Z[batch*2+idx], Z[batch*3+idx],
        min_a, min_c, max_ab, max_cd, Ms);
  };
  cta_launch<launch_t>(K, num_ctas, context);
}

}  // namespace gpu
}  // namespace plas

#endif
