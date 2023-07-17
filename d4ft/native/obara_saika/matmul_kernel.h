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

#ifndef PLAS_GPU_MATMUL_KERNEL_H_
#define PLAS_GPU_MATMUL_KERNEL_H_

#include "../memory.h"
#include "../loadstore.h"
#include "../kernel_launch.h"

namespace plas {
namespace gpu {

template <typename input_it, typename output_it>
void matmul_baseline(input_it A, input_it B, int m, int n, int k, output_it C, context_t& context) {

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  auto K = [=] PLAS_DEVICE(int tid) {
    size_t mm = tid % m;
    size_t nn = tid / m;
    type_t tmp = 0;
    for (int i = 0; i < k; ++i) {
      tmp += ldg(A + (mm * k + i)) * ldg(B + (i * n + nn));
    }
    C[mm * n + nn] = tmp;
  };
  transform<1024, 1>(K, m*n, context);
}

template <typename input_it, typename output_it>
void matmul_coalesce(input_it A, input_it B, int m, int n, int k, output_it C, context_t& context) {

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  auto K = [=] PLAS_DEVICE(int tid) {
    size_t mm = tid / n;
    size_t nn = tid % n;
    type_t tmp = 0;
    for (int i = 0; i < k; ++i) {
      tmp += ldg(A + (mm * k + i)) * ldg(B + (i * n + nn));
    }
    C[mm * n + nn] = tmp;
  };
  transform<1024, 1>(K, m*n, context);
}

template <typename input_it, typename output_it>
void matmul_shmem(input_it A, input_it B, int m, int n, int k, output_it C, context_t& context) {
  const int kBlockSize = 32;
  typedef launch_params_t<1024, 1> launch_t;
  typedef typename std::iterator_traits<input_it>::value_type type_t;
  int A_block_size = div_up(m, kBlockSize);
  int B_block_size = div_up(n, kBlockSize);
  int num_ctas = A_block_size * B_block_size;

  auto K = [=] PLAS_DEVICE(int tid, int cta) {
    __shared__ type_t As[kBlockSize*kBlockSize];
    __shared__ type_t Bs[kBlockSize*kBlockSize];

    int A_block_idx = cta / B_block_size;
    int B_block_idx = cta % B_block_size;
    int A_block_start_row = A_block_idx * kBlockSize;
    int B_block_start_col = B_block_idx * kBlockSize;
    int C_start = A_block_idx * n * kBlockSize + B_block_idx * kBlockSize;
    int A_global_offset = A_block_start_row * k;
    int B_global_offset = B_block_start_col;
    int C_global_offset = C_start;

    int thread_row_idx = tid / kBlockSize;
    int thread_col_idx = tid % kBlockSize;

    type_t tmp = 0;
    //compute per thread starting idx at A and B
    for (int k_idx = 0; k_idx < k; k_idx += kBlockSize) { 

      // load 32*32 from A to As
      As[thread_row_idx * kBlockSize + thread_col_idx] = ldg(A + A_global_offset + thread_row_idx * k + thread_col_idx);
      Bs[thread_row_idx * kBlockSize + thread_col_idx] = ldg(B + B_global_offset + thread_row_idx * n + thread_col_idx);

      __syncthreads();
      A_global_offset += kBlockSize;
      B_global_offset += kBlockSize * n; 

      for (int acc_idx = 0; acc_idx < kBlockSize; ++acc_idx) {
        tmp += As[thread_row_idx * kBlockSize + acc_idx] * Bs[acc_idx * kBlockSize + thread_col_idx];
      }

      __syncthreads();
    }
    C[C_global_offset + thread_row_idx * n + thread_col_idx] = tmp;
  };
  cta_launch<launch_t>(K, num_ctas, context);
}

template <typename input_it, typename output_it>
void matmul_1dblocktiling(input_it A, input_it B, int m, int n, int k, output_it C, context_t& context) {
  const int kBlockSize = 64;
  const int kBK = 8;
  typedef launch_params_t<512, 1> launch_t;
  typedef typename std::iterator_traits<input_it>::value_type type_t;
  int A_block_size = div_up(m, kBlockSize);
  int B_block_size = div_up(n, kBlockSize);
  // we shape it as [B_block_size, A_block_size]
  int num_ctas = A_block_size * B_block_size;

  auto K = [=] PLAS_DEVICE(int tid, int cta) {
    __shared__ type_t As[kBlockSize * kBK];
    __shared__ type_t Bs[kBlockSize * kBK];

    int A_block_idx = cta / B_block_size;
    int B_block_idx = cta % B_block_size;

    int A_global_offset = A_block_idx * kBlockSize * k;
    int B_global_offset = B_block_idx * kBlockSize;
    int C_global_offset = A_block_idx * n * kBlockSize + B_block_idx * kBlockSize;

    int A_thread_row_idx = tid / kBK;
    int A_thread_col_idx = tid % kBK;
    
    int B_thread_row_idx = tid / kBlockSize;
    int B_thread_col_idx = tid % kBlockSize;

    int global_thread_row_idx = tid / kBlockSize;
    int global_thread_col_idx = tid % kBlockSize;

    type_t tmp[kBK] = {0};
    //compute per thread starting idx at A and B
    for (int k_idx = 0; k_idx < k; k_idx += kBK) { 

      // load 64*8 from A to As, 8*64 from B to Bs
      As[A_thread_row_idx * kBK + A_thread_col_idx] = ldg(A + A_global_offset + A_thread_row_idx * k + A_thread_col_idx);
      Bs[B_thread_row_idx * kBlockSize + B_thread_col_idx] = ldg(B + B_global_offset + B_thread_row_idx * n + B_thread_col_idx); 

      __syncthreads();
      A_global_offset += kBK;
      B_global_offset += kBK * n;
      for (int res_idx = 0; res_idx < kBK; ++res_idx) {
        for (int acc_idx = 0; acc_idx < kBK; ++acc_idx) {
          tmp[res_idx] += As[(global_thread_row_idx*kBK+res_idx) * kBK + acc_idx] * Bs[acc_idx * kBlockSize + global_thread_col_idx];
        }
      }

      __syncthreads();
    }
    for (int res_idx = 0; res_idx < kBK; ++res_idx) {
      int cc = C_global_offset + (global_thread_row_idx * kBK + res_idx) * n + global_thread_col_idx;
      C[cc] = tmp[res_idx];
    }
  };
  cta_launch<launch_t>(K, num_ctas, context);
}

template <typename input_it, typename output_it>
void matmul_2dblocktiling(input_it A, input_it B, int m, int n, int k, output_it C, context_t& context) {
  const int kBlockSize = 64;
  const int kBK = 8;
  typedef launch_params_t<64, 1> launch_t;
  typedef typename std::iterator_traits<input_it>::value_type type_t;
  int A_block_size = div_up(m, kBlockSize);
  int B_block_size = div_up(n, kBlockSize);
  // we shape it as [B_block_size, A_block_size]
  int num_ctas = A_block_size * B_block_size;

  auto K = [=] PLAS_DEVICE(int tid, int cta) {
    __shared__ type_t As[kBlockSize * kBK];
    __shared__ type_t Bs[kBlockSize * kBK];

    int A_block_idx = cta / B_block_size;
    int B_block_idx = cta % B_block_size;

    int A_global_offset = A_block_idx * kBlockSize * k;
    int B_global_offset = B_block_idx * kBlockSize;
    int C_global_offset = A_block_idx * n * kBlockSize + B_block_idx * kBlockSize;

    int A_thread_row_idx = tid / kBK;
    int A_thread_col_idx = tid % kBK;

    int B_thread_row_idx = tid / kBlockSize;
    int B_thread_col_idx = tid % kBlockSize;

    int global_thread_row_idx = tid / kBK;
    int global_thread_col_idx = tid % kBK;

    int A_stride = kBlockSize / kBK;
    int B_stride = 1;

    type_t tmp[kBK*kBK] = {0};

    type_t reg_A[kBK] = {0};
    type_t reg_B[kBK] = {0};

    //compute per thread starting idx at A and B
    for (int k_idx = 0; k_idx < k; k_idx += kBK) {

      // load 64*8 from A to As, 8*64 from B to Bs
      for (size_t load_offset = 0; load_offset < kBlockSize; load_offset += A_stride) {
        As[(A_thread_row_idx + load_offset) * kBK + A_thread_col_idx] = ldg(A + A_global_offset + (A_thread_row_idx + load_offset) * k + A_thread_col_idx);
      }

      for (size_t load_offset = 0; load_offset < kBK; load_offset += B_stride) {
        Bs[(B_thread_row_idx + load_offset) * kBlockSize + B_thread_col_idx] = ldg(B + B_global_offset + (B_thread_row_idx + load_offset) * n + B_thread_col_idx);
      }

      __syncthreads();
      A_global_offset += kBK;
      B_global_offset += kBK * n;

      for (int acc_idx = 0; acc_idx < kBK; ++acc_idx) {
        // block into registers
        for (int res_idx = 0; res_idx < kBK; ++res_idx) {
          reg_A[res_idx] = As[(global_thread_row_idx*kBK+res_idx) * kBK + acc_idx];
          reg_B[res_idx] = Bs[acc_idx * kBlockSize + global_thread_col_idx * kBK + res_idx];
        }
        for (int res_idx_A = 0; res_idx_A < kBK; ++res_idx_A) {
          for (int res_idx_B = 0; res_idx_B < kBK; ++res_idx_B) {
            tmp[res_idx_A * kBK + res_idx_B] += reg_A[res_idx_A] * reg_B[res_idx_B];
          }
        }
      }
      __syncthreads();
    }

    for (int res_idx_A = 0; res_idx_A < kBK; ++res_idx_A) {
      for (int res_idx_B = 0; res_idx_B < kBK; ++res_idx_B) {
        int cc = C_global_offset + (global_thread_row_idx * kBK + res_idx_A) * n + global_thread_col_idx * kBK + res_idx_B;
        C[cc] = tmp[res_idx_A * kBK + res_idx_B];
      }
    }
  };
  cta_launch<launch_t>(K, num_ctas, context);
}

template <typename input_it, typename output_it>
void matmul_2dblocktiling_vec(input_it A, input_it B, int m, int n, int k, output_it C, context_t& context) {
  const int kBlockSize = 128;
  const int kBK = 8;
  typedef launch_params_t<256, 1> launch_t;
  typedef typename std::iterator_traits<input_it>::value_type type_t;
  int A_block_size = div_up(m, kBlockSize);
  int B_block_size = div_up(n, kBlockSize);
  // we shape it as [B_block_size, A_block_size]
  int num_ctas = A_block_size * B_block_size;

  auto K = [=] PLAS_DEVICE(int tid, int cta) {
    __shared__ type_t As[kBlockSize * kBK];
    __shared__ type_t Bs[kBlockSize * kBK];

    int A_block_idx = cta / B_block_size;
    int B_block_idx = cta % B_block_size;

    int A_global_offset = A_block_idx * kBlockSize * k;
    int B_global_offset = B_block_idx * kBlockSize;
    int C_global_offset = A_block_idx * n * kBlockSize + B_block_idx * kBlockSize;

    int A_thread_row_idx = tid / (kBK/4);
    int A_thread_col_idx = tid % (kBK/4);

    int B_thread_row_idx = tid / (kBlockSize/4);
    int B_thread_col_idx = tid % (kBlockSize/4);

    int global_thread_row_idx = tid / (kBlockSize/kBK);
    int global_thread_col_idx = tid % (kBlockSize/kBK);

    type_t tmp_res[kBK*kBK] = {0};

    type_t reg_A[kBK] = {0};
    type_t reg_B[kBK] = {0};

    //compute per thread starting idx at A and B
    for (int k_idx = 0; k_idx < k; k_idx += kBK) {

      // populate shmem with 64*8 from A and 8*64 from B (transpose A)
      float4 tmp_A = reinterpret_cast<float4*>(&A[A_global_offset + A_thread_row_idx * k + A_thread_col_idx * 4])[0];
      As[(A_thread_col_idx * 4 + 0) * kBlockSize + A_thread_row_idx] = tmp_A.x;
      As[(A_thread_col_idx * 4 + 1) * kBlockSize + A_thread_row_idx] = tmp_A.y;
      As[(A_thread_col_idx * 4 + 2) * kBlockSize + A_thread_row_idx] = tmp_A.z;
      As[(A_thread_col_idx * 4 + 3) * kBlockSize + A_thread_row_idx] = tmp_A.w;

      /*reinterpret_cast<float4*>
        (&Bs[B_thread_row_idx * kBlockSize + B_thread_col_idx * 4])[0] =
          reinterpret_cast<float4*>(&B[B_global_offset + B_thread_row_idx * n + B_thread_col_idx * 4])[0];*/
      float4 tmp_B = reinterpret_cast<float4*>
        (&B[B_global_offset + B_thread_row_idx * n + B_thread_col_idx * 4])[0];
      Bs[((B_thread_col_idx % 2) * 4 + B_thread_row_idx * 8 + 0) * 16 + B_thread_col_idx / 2] = tmp_B.x;
      Bs[((B_thread_col_idx % 2) * 4 + B_thread_row_idx * 8 + 1) * 16 + B_thread_col_idx / 2] = tmp_B.y;
      Bs[((B_thread_col_idx % 2) * 4 + B_thread_row_idx * 8 + 2) * 16 + B_thread_col_idx / 2] = tmp_B.z;
      Bs[((B_thread_col_idx % 2) * 4 + B_thread_row_idx * 8 + 3) * 16 + B_thread_col_idx / 2] = tmp_B.w;
      __syncthreads();

      A_global_offset += kBK;
      B_global_offset += kBK * n;

      for (int acc_idx = 0; acc_idx < kBK; ++acc_idx) {
        // block into registers
        for (int res_idx = 0; res_idx < kBK; ++res_idx) {
          reg_A[res_idx] = As[acc_idx * kBlockSize + global_thread_row_idx * kBK + res_idx];
          //reg_B[res_idx] = Bs[acc_idx * kBlockSize + global_thread_col_idx * kBK + res_idx];
          reg_B[res_idx] = Bs[(acc_idx * 8 + res_idx) * 16 + global_thread_col_idx];
        }
        for (int res_idx_A = 0; res_idx_A < kBK; ++res_idx_A) {
          for (int res_idx_B = 0; res_idx_B < kBK; ++res_idx_B) {
            tmp_res[res_idx_A * kBK + res_idx_B] += reg_A[res_idx_A] * reg_B[res_idx_B];
          }
        }
      }
      __syncthreads();
    }

    for (int res_idx_A = 0; res_idx_A < kBK; ++res_idx_A) {
      for (int res_idx_B = 0; res_idx_B < kBK; res_idx_B += 4) {
        float4 tmp_C = reinterpret_cast<float4*>
          (&C[C_global_offset +
          (global_thread_row_idx * kBK + res_idx_A) * n +
           global_thread_col_idx * kBK + res_idx_B])[0];
        tmp_C.x = tmp_res[res_idx_A * kBK + res_idx_B];
        tmp_C.y = tmp_res[res_idx_A * kBK + res_idx_B + 1];
        tmp_C.z = tmp_res[res_idx_A * kBK + res_idx_B + 2];
        tmp_C.w = tmp_res[res_idx_A * kBK + res_idx_B + 3];
        reinterpret_cast<float4*>
          (&C[C_global_offset +
          (global_thread_row_idx * kBK + res_idx_A) * n +
           global_thread_col_idx * kBK + res_idx_B])[0] = tmp_C;
      }
    }
  };
  cta_launch<launch_t>(K, num_ctas, context);
}

}  // namespace gpu
}  // namespace plas

#endif