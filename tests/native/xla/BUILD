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

# load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "example_h",
    hdrs = ["example.h"],
    copts = [
        "--std=c++17",
    ],
    deps = [
        "//d4ft/native/xla:specs",
    ],
)

# cuda_library(
#     name = "example_cu",
#     srcs = ["example.cu"],
#     copts = [
#         "--std=c++17",
#     ],
#     deps = [
#         ":example_h",
#         # "@cuda//:cudart_static",
#     ],
# )

# pybind_extension(
#     name = "example",
#     srcs = [
#         "example.cc",
#     ],
#     copts = [
#         "--std=c++17",
#     ],
#     deps = [
#         ":example_cu",
#         "//d4ft/native/xla:custom_call_h",
#     ],
# )

# py_test(
#     name = "example_test",
#     srcs = ["example_test.py"],
#     data = [
#         ":example.so",
#         # "@cuda//:bin",
#     ],
#     deps = [
#         "//d4ft/native/xla:custom_call",
#         requirement("absl-py"),
#         requirement("jax"),
#         requirement("jaxlib"),
#     ],
# )
