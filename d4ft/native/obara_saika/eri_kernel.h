#ifndef PLAS_GPU_ERI_KERNEL_H_
#define PLAS_GPU_ERI_KERNEL_H_

#include "hemi/hemi.h"
#include <cmath>

template <typename FLOAT>
extern void hartree(const size_t N, const size_t *n, const FLOAT *r,
                    const FLOAT *z, const size_t *min_a, const size_t *min_c,
                    const size_t *max_ab, const size_t *max_cd,
                    const size_t *Ms, FLOAT *output);

#endif
