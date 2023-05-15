#include <iostream>
#include "eri.h"
#include "hemi/hemi.h"
#include "hemi/parallel_for.h"
#include "hemi/device_api.h"
#include "hemi/array.h"

void ConstantInit(int* ptr, int size, int constant) {
  for (int i = 0; i < size; ++i) {
    ptr[i] = constant;
  }
}

int main() {
  int N = 10;
  hemi::Array<int> n(12 * N, true);
  for (int i = 0; i < 12 * N; ++i) {
    n.writeOnlyHostPtr()[i] = rand() % 2;
  }
  hemi::Array<int> min_a(3, true);
  hemi::Array<int> min_c(3, true);
  hemi::Array<int> max_ab(3, true);
  hemi::Array<int> max_cd(3, true);
  hemi::Array<int> Ms(3, true);
  ConstantInit(min_a.writeOnlyHostPtr(), 3, 0);
  ConstantInit(min_c.writeOnlyHostPtr(), 3, 0);
  ConstantInit(max_ab.writeOnlyHostPtr(), 3, 0);
  ConstantInit(max_cd.writeOnlyHostPtr(), 3, 0);

  typedef int Ang[12][N];
  Ang angular = static_cast<Ang>(n.readOnlyHostPtr());

  // min a, c
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < N; ++j) {
      min_a.hostPtr()[i] = std::min(
          min_a.hostPtr()[i], angular[i][j]);
    }
  }
  for (int i = 6; i < 9; ++i) {
    for (int j = 0; j < N; ++j) {
      min_c.hostPtr()[i - 6] = std::min(
          min_c.hostPtr()[i - 6], angular[i][j])
    }
  }

  // max ab, cd
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < N; ++j) {
      max_ab.hostPtr()[i] = std::max(
          max_ab.hostPtr()[i], angular[i][j] +
          angular[i + 3][j]);
    }
  }
  for (int i = 6; i < 9; ++i) {
    for (int j = 0; j < N; ++j) {
      max_cd.hostPtr()[i - 6] = std::max(
          max_cd.hostPtr()[i - 6], angular[i][j] +
          angular[(i + 3)][j]);
    }
  }

  // Ms
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < N; ++j) {
      int s = 0;
      for (int k = i; k < 3; ++k) {
        for (int l = 0; l < 4; ++l) {
          s += angular[l * 3 + k][j];
        }
      }
      Ms.hostPtr()[i] = std::max(Ms.hostPtr()[i], s);
  }

  hemi::parallel_for(0, N, [&] HEMI_LAMBDA(int i) {
    eri<float>();
    int* ni = n.writeOnlyPtr() + i * 12;
    int* na = ni;
    int* nb = ni + 3;
    int* nc = ni + 6;
    int* nd = ni + 9;
    int* min_a = ni + 12;
    int* min_c = ni + 15;
    int* max_ab = ni + 18;
    int* max_cd = ni + 21;
    int* Ms = ni + 24;
  });

  int* ms = Ms.writeOnlyHostPtr();
  ms[0] = 12;
  ms[1] = 8;
  ms[2] = 4;
  hemi::parallel_for(0, )

  float out = plas::gpu::eri<float>(
      1, 1, 1, // na.{x, y, z}
      1, 1, 1, // nb.{x, y, z}
      1, 1, 1, // nc.{x, y, z}
      1, 1, 1, // nd.{x, y, z}
      0., 1., 1., // ra.{x, y, z}
      0., 1., 0., // rb.{x, y, z}
      1., 1., 0., // rc.{x, y, z}
      0., 0., 0., // rd.{x, y, z}
      2., 1.5, 1.3, 1.2, // za, zb, zc, zd
      min_a, min_c, max_ab, max_cd, Ms);
  std::cout << out << std::endl;
}
