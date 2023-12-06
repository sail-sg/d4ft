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

#ifndef PLAS_GPU_ERI_KERNEL_H_
#define PLAS_GPU_ERI_KERNEL_H_

#include <cuda_runtime_api.h>

#include <cstring>

#include <cmath>

#include "hemi/hemi.h"

#include "d4ft/native/xla/specs.h"

class Hartree_32 {
 public:
  // template <typename FLOAT>
  static auto ShapeInference(const Spec<int>& shape1,
                             const Spec<int>& shape10,
                             const Spec<int>& shape2,
                             const Spec<float>& shape3,
                             const Spec<float>& shape4,
                             const Spec<int>& shape5,
                             const Spec<int>& shape6,
                             const Spec<int>& shape7,
                             const Spec<int>& shape8,
                             const Spec<int>& shape9,
                             const Spec<int>& shape11) {
    // float n2 = shape4.shape[0]*(shape4.shape[0]+1)/2;
    // float n4 = n2*(n2+1)/2;
    // int n4_int = static_cast<int>(n4);
    std::vector<int> outshape={shape1.shape[0]};
    Spec<float> out(outshape);
    return std::make_tuple(out);
  }
  // static void Cpu(Array<const float>& arg1, Array<const int>& arg2,
  //                 Array<float>& out) {
  //   std::memcpy(out.ptr, arg1.ptr, sizeof(float) * arg1.spec->Size());
  // }
  // template <typename FLOAT>
  static void Cpu(Array<const int>& N, 
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
                  Array<float>& output){
      // std::memcpy(output.ptr, outshape.ptr, sizeof(float) * outshape.spec->Size());
  }

  // template <typename FLOAT>
  static void Gpu(cudaStream_t stream, 
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
                  Array<float>& output);
};

class Hartree_64 {
 public:
  // template <typename FLOAT>
  static auto ShapeInference(const Spec<int>& shape1,
                             const Spec<int>& shape21,
                             const Spec<int64_t>& shape10,
                             const Spec<int64_t>& shape22,
                             const Spec<int>& shape2,
                             const Spec<double>& shape3,
                             const Spec<double>& shape4,
                             const Spec<int>& shape5,
                             const Spec<int>& shape6,
                             const Spec<int>& shape7,
                             const Spec<int>& shape8,
                             const Spec<int>& shape9,
                             const Spec<int>& shape11,
                             const Spec<int>& shape12,
                             const Spec<int>& shape13,
                            //  const Spec<int>& shape14,
                             const Spec<int>& shape23,
                             const Spec<int>& shape24,
                             const Spec<double>& shape15,
                             const Spec<double>& shape16,
                             const Spec<int>& shape17,
                             const Spec<double>& shape18,
                             const Spec<int>& shape19,
                             const Spec<int>& shape20) {
    // double n2 = shape4.shape[0]*(shape4.shape[0]+1)/2;
    // double n4 = n2*(n2+1)/2;
    // int n4_int = static_cast<int>(n4);
    // std::vector<int> outshape={2*shape11.shape[0]*shape12.shape[0]};
    std::vector<int> outshape={shape1.shape[0]};
    Spec<double> out(outshape);
    return std::make_tuple(out);
  }
  // static void Cpu(Array<const float>& arg1, Array<const int>& arg2,
  //                 Array<float>& out) {
  //   std::memcpy(out.ptr, arg1.ptr, sizeof(float) * arg1.spec->Size());
  // }
  // template <typename FLOAT>
  static void Cpu(Array<const int>& N,
                  Array<const int>& thread_load, 
                  Array<const int64_t>& thread_num,
                  Array<const int64_t>& screened_length, 
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
                  // Array<const int>& screened_idx_offset,
                  Array<const int>& ab_thread_num,
                  Array<const int>& ab_thread_offset,
                  Array<const double>& pgto_coeff,
                  Array<const double>& pgto_normalization_factor,
                  Array<const int>& pgto_idx_to_cgto_idx,
                  Array<const double>& rdm1,
                  Array<const int>& n_cgto,
                  Array<const int>& n_pgto,
                  Array<double>& output){
      // std::memcpy(output.ptr, outshape.ptr, sizeof(float) * outshape.spec->Size());
  }

  // template <typename FLOAT>
  static void Gpu(cudaStream_t stream, 
                  Array<const int>& N,
                  Array<const int>& thread_load, 
                  Array<const int64_t>& thread_num,
                  Array<const int64_t>& screened_length, 
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
                  // Array<const int>& screened_idx_offset,
                  Array<const int>& ab_thread_num,
                  Array<const int>& ab_thread_offset,
                  Array<const double>& pgto_coeff,
                  Array<const double>& pgto_normalization_factor,
                  Array<const int>& pgto_idx_to_cgto_idx,
                  Array<const double>& rdm1,
                  Array<const int>& n_cgto,
                  Array<const int>& n_pgto,
                  Array<double>& output);
};

class Hartree_32_uncontracted {
 public:
  // template <typename FLOAT>
  static auto ShapeInference(const Spec<int>& shape1,
                             const Spec<int>& shape10,
                             const Spec<int>& shape2,
                             const Spec<float>& shape3,
                             const Spec<float>& shape4,
                             const Spec<int>& shape5,
                             const Spec<int>& shape6,
                             const Spec<int>& shape7,
                             const Spec<int>& shape8,
                             const Spec<int>& shape9) {
    // float n2 = shape4.shape[0]*(shape4.shape[0]+1)/2;
    // float n4 = n2*(n2+1)/2;
    // int n4_int = static_cast<int>(n4);
    std::vector<int> outshape={shape10.shape[0]};
    Spec<float> out(outshape);
    return std::make_tuple(out);
  }
  // static void Cpu(Array<const float>& arg1, Array<const int>& arg2,
  //                 Array<float>& out) {
  //   std::memcpy(out.ptr, arg1.ptr, sizeof(float) * arg1.spec->Size());
  // }
  // template <typename FLOAT>
  static void Cpu(Array<const int>& N, 
                  Array<const int>& index_4c,    
                  Array<const int>& n, 
                  Array<const float>& r, 
                  Array<const float>& z,
                  Array<const int>& min_a, 
                  Array<const int>& min_c, 
                  Array<const int>& max_ab,
                  Array<const int>& max_cd, 
                  Array<const int>& Ms, 
                  Array<float>& output){
      // std::memcpy(output.ptr, outshape.ptr, sizeof(float) * outshape.spec->Size());
  }

  // template <typename FLOAT>
  static void Gpu(cudaStream_t stream, 
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
                  Array<float>& output);
};

class Hartree_64_uncontracted {
 public:
  // template <typename FLOAT>
  static auto ShapeInference(const Spec<int>& shape1,
                             const Spec<int>& shape10,
                             const Spec<int>& shape2,
                             const Spec<double>& shape3,
                             const Spec<double>& shape4,
                             const Spec<int>& shape5,
                             const Spec<int>& shape6,
                             const Spec<int>& shape7,
                             const Spec<int>& shape8,
                             const Spec<int>& shape9) {
    // double n2 = shape4.shape[0]*(shape4.shape[0]+1)/2;
    // double n4 = n2*(n2+1)/2;
    // int n4_int = static_cast<int>(n4);
    std::vector<int> outshape={shape10.shape[0]};
    Spec<double> out(outshape);
    return std::make_tuple(out);
  }
  // static void Cpu(Array<const float>& arg1, Array<const int>& arg2,
  //                 Array<float>& out) {
  //   std::memcpy(out.ptr, arg1.ptr, sizeof(float) * arg1.spec->Size());
  // }
  // template <typename FLOAT>
  static void Cpu(Array<const int>& N, 
                  Array<const int>& index_4c,  
                  Array<const int>& n, 
                  Array<const double>& r, 
                  Array<const double>& z,
                  Array<const int>& min_a, 
                  Array<const int>& min_c, 
                  Array<const int>& max_ab,
                  Array<const int>& max_cd, 
                  Array<const int>& Ms, 
                  Array<double>& output){
      // std::memcpy(output.ptr, outshape.ptr, sizeof(float) * outshape.spec->Size());
  }

  // template <typename FLOAT>
  static void Gpu(cudaStream_t stream, 
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
                  Array<double>& output);
};

#endif
