#include <cuda_runtime_api.h>
#include "example.h"

void ExampleMember::Gpu(cudaStream_t stream, Array<const float>& arg1,
                        Array<const int>& arg2, Array<float>& out) {
  cudaMemcpy(out.ptr, arg1.ptr, sizeof(float) * arg1.spec->Size(), cudaMemcpyDeviceToDevice);
}

void Example::Gpu(cudaStream_t stream, Array<const float>& arg1,
                  Array<const int>& arg2, Array<float>& out) {
  cudaMemcpy(out.ptr, arg1.ptr, sizeof(float) * arg1.spec->Size(), cudaMemcpyDeviceToDevice);
}
