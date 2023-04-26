#include <iostream>
#include "eri.h"

int main() {
  int min_a[3] = {1, 1, 1};
  int min_c[3] = {1, 1, 1};
  int max_ab[3] = {2, 2, 2};
  int max_cd[3] = {2, 2, 2};
  int Ms[3] = {12, 8, 4};
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
