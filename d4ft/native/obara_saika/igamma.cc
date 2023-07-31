// Copyright 2023 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>

#include <array>
#include <iostream>

#define FLOAT double

int main() {
  std::cout << Lgamma(1.) << std::endl;
  std::cout << Digamma(1.) << std::endl;
  std::cout << IgammaSeries<VALUE>(1., 1., 1., true) << std::endl;
  std::cout << IgammacContinuedFraction<VALUE>(1., 1., 1., true) << std::endl;
  std::cout << Igamma(1.2, 1.2) << std::endl;
  std::cout << Igammac(1.2, 1.2) << std::endl;
}
