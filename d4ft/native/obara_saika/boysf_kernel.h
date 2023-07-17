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

#ifndef PLAS_GPU_BOYSF_KERNEL_H_
#define PLAS_GPU_BOYSF_KERNEL_H_

#include "../reduce_cta.h"
#include "../memory.h"
#include "../loadstore.h"
#include "../kernel_launch.h"

namespace plas {
namespace gpu {

template <typename launch_arg_t = empty_t, typename input_it,
          typename output_it>
void boysf(input_it input, int n, int count, output_it output, context_t& context) {

  typedef typename conditional_typedef_t<
      launch_arg_t, launch_params_t<512, 13> >::type_t launch_t;
  typedef typename std::iterator_traits<input_it>::value_type type_t;
  int num_ctas = launch_t::cta_dim(context).num_ctas(count);

  auto k = [=] PLAS_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      nv = nt * vt
    };
    typedef cta_reduce_t<nt, type_t> reduce_t;
    __shared__ typename reduce_t::storage_t shared_reduce;

    int ngrids[4] = {100, 400, 1600, 6400};
    type_t tdts[4];
    type_t tresults[4];

    for (int idx = 0; idx < 4; ++idx) {
        type_t inv_ngrid = 1.0/ngrids[idx];
        int n2 = 2*n;
        type_t tresult = (n2 == 0) ? 0.5 : 0;
        array_t<type_t, vt> val;
        strided_iterate<nt, vt>([&](int i, int j) {
            type_t t = (j+1)*inv_ngrid;
            val[i] = exp(-input[cta]*t*t)*pow(t, n2);
        }, tid, ngrids[idx]);

        // reduce val[vt] into scalar
        type_t scalar;
        plus_t<type_t> op;
        strided_iterate<nt, vt>([&](int i, int j) {
            scalar = i ? op(scalar, val[i]) : val[0];
        }, tid, ngrids[idx]);

        // reduce across all threads
        scalar = reduce_t().reduce(tid, scalar, shared_reduce,
            min(ngrids[idx], (int)nt), op, false);

        if (!tid) {
            tresult += (scalar + exp(-input[cta])/2);
            tresult *= inv_ngrid;
            tdts[idx] = inv_ngrid;
            tresults[idx] = tresult;
        }
    }

    // thread 0 of each cta does the extrapolation
    if (!tid) {
        type_t tresult01 = (-tdts[1])*(tresults[0]-tresults[1])/(tdts[0]-tdts[1])+tresults[1];
        type_t tresult12 = (-tdts[2])*(tresults[1]-tresults[2])/(tdts[1]-tdts[2])+tresults[2];
        type_t tresult23 = (-tdts[3])*(tresults[2]-tresults[3])/(tdts[2]-tdts[3])+tresults[3];
        type_t tresult012 = (-tdts[2])*(tresult01-tresult12)/(tdts[0]-tdts[2])+tresult12;
        type_t tresult123 = (-tdts[3])*(tresult12-tresult23)/(tdts[1]-tdts[3])+tresult23;
        type_t result = (-tdts[3])*(tresult012-tresult123)/(tdts[0]-tdts[3])+tresult123;
        output[cta] = result;
    }
  };
    
  cta_launch<launch_t>(k, num_ctas, context);
}

}  // namespace gpu
}  // namespace plas

#endif