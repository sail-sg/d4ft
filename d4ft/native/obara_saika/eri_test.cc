#include "eri_kernel.h"
#include "hemi/array.h"
#include "hemi/device_api.h"
#include "hemi/hemi.h"
#include "hemi/parallel_for.h"
#include <iostream>

int main() {
  int N = 10;
  hemi::Array<int> n(3 * N, true);
  for (int i = 0; i < 3 * N; ++i) {
    n.writeOnlyHostPtr()[i] = rand() % 2;
  }
  hemi::Array<float> r(3 * N, true);
  hemi::Array<float> z(N, true);
  for (int i = 0; i < 3 * N; ++i) {
    r.writeOnlyHostPtr()[i] =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < N; ++i) {
    z.writeOnlyHostPtr()[i] =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  hemi::Array<int> min_(3, true);
  hemi::Array<int> max_(3, true);
  hemi::Array<int> max_ab(3, true);
  hemi::Array<int> max_cd(3, true);
  hemi::Array<int> Ms(3, true);
  constant_init(min_a.writeOnlyHostPtr(), 3, 0);
  constant_init(min_c.writeOnlyHostPtr(), 3, 0);
  constant_init(max_ab.writeOnlyHostPtr(), 3, 0);
  constant_init(max_cd.writeOnlyHostPtr(), 3, 0);

  typedef int Ang[3][N];
  Ang angular = static_cast<Ang>(n.readOnlyHostPtr());

  // min a, c
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < N; ++j) {
      min_.hostPtr()[i] = std::min(min_.hostPtr()[i], angular[i][j]);
      max_.hostPtr()[i] = std::max(max_.hostPtr()[i], angular[i][j]);
    }
  }

  // max ab, cd
  for (int i = 0; i < 3; ++i) {
    max_ab.hostPtr()[i] = max_.hostPtr()[i] * 2;
    max_cd.hostPtr()[i] = max_.hostPtr()[i] * 2;
  }

  // Ms
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < N; ++j) {
      int s = 0;
      for (int k = i; k < 3; ++k) {
        s += angular[k][j];
      }
      Ms.hostPtr()[i] = std::max(Ms.hostPtr()[i], s);
    }
    Ms.hostPtr()[i] = Ms.hostPtr()[i] * 4;
  }
  hartree<float>(N, n.readOnlyPtr(), r.readOnlyPtr(), z.readOnlyPtr(),
                 min_.readOnlyPtr(), min_.readOnlyPtr(), max_ab.readOnlyPtr(),
                 max_cd.readOnlyPtr(), Ms.readOnlyPtr(), nullptr);
}
