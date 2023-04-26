#include <math.h>
#include <iostream>
#include <array>

#define FLOAT double

int main() {
  std::cout << Lgamma(1.) << std::endl;
  std::cout << Digamma(1.) << std::endl;
  std::cout << IgammaSeries<VALUE>(1., 1., 1., true) << std::endl;
  std::cout << IgammacContinuedFraction<VALUE>(1., 1., 1., true) << std::endl;
  std::cout << Igamma(1.2, 1.2) << std::endl;
  std::cout << Igammac(1.2, 1.2) << std::endl;
}
