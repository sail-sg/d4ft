#ifndef PLAS_GPU_ERI_KERNEL_H_
#define PLAS_GPU_ERI_KERNEL_H_

template <typename FLOAT>
void hartree(const int N, const FLOAT *n, const FLOAT *r, const FLOAT *z,
             const int *min_a, const int *min_c, const int *max_ab,
             const int *max_cd, const int *Ms, FLOAT *output);

#endif
