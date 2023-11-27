#include <cuda_runtime_api.h>

#include <cstring>

#include "d4ft/native/xla/specs.h"

class Example {
 public:
  static auto ShapeInference(const Spec<float>& shape1,
                             const Spec<int>& shape2) {
    return std::make_tuple(shape1);
  }
  static void Cpu(Array<const float>& arg1, Array<const int>& arg2,
                  Array<float>& out) {
    std::memcpy(out.ptr, arg1.ptr, sizeof(float) * arg1.spec->Size());
  }
  static void Gpu(cudaStream_t stream, Array<const float>& arg1,
                  Array<const int>& arg2, Array<float>& out);
};

class Parent {
 public:
  int i = 0;
};

class ExampleMember {
 public:
  Parent* p_;
  ExampleMember(Parent* p) : p_(p) {}
  auto ShapeInference(const Spec<float>& shape1, const Spec<int>& shape2) {
    return std::make_tuple(shape1);
  }
  void Cpu(Array<const float>& arg1, Array<const int>& arg2,
           Array<float>& out) {
    std::memcpy(out.ptr, arg1.ptr, sizeof(float) * arg1.spec->Size());
  }
  void Gpu(cudaStream_t stream, Array<const float>& arg1,
           Array<const int>& arg2, Array<float>& out);
};
