# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module provides template for defining XLA custom call.

To define a new custom call using this module, you need to:
1. Define a class which contains 3 static methods: ShapeInference (for abstract
  evaluation), Cpu and Gpu (the actual ops, where the Cpu method dispatches CUDA
  kernels)
2. Pass the class to the REGISTER_XLA_FUNCTION or REGISTER_XLA_MEMBER macro.
  The macro then creates a subclass with Xla supports and makes it a pybind
  module.
"""
