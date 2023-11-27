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

load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

# cc_library(
#     name = "eri_kernel_h",
#     hdrs = ["eri_kernel.h"],
#     copts = [
#         "--std=c++17",
#     ],
#     deps = [
#         "@hemi",
#         "//d4ft/native/xla:specs",
#     ],
# )


# cc_library(
#     name = "boys_h",
#     hdrs = ["boys.h"],
#     copts = [
#         "--std=c++17",
#     ],
#     deps = [
#         "@hemi",
#         "//d4ft/native/gamma:igamma",
#         "//d4ft/native/gamma:lgamma",
#     ],
# )


# cc_library(
#     name = "comb_h",
#     hdrs = ["comb.h"],
#     copts = [
#         "--std=c++17",
#     ],
#     deps = [
#         "@hemi",
#     ],
# )

# cc_library(
#     name = "eri_h",
#     hdrs = ["eri.h"],
#     copts = [
#         "--std=c++17",
#     ],
#     deps = [
#         "@hemi",
#         "comb_h",
#         "boys_h",
#     ],
# )

# cuda_library(
#     name = "eri_kernel_cu",
#     srcs = ["eri_kernel.cu"],
#     copts = [
#         "--expt-extended-lambda",
#         "--expt-relaxed-constexpr",
#     ],
#     deps = [
#         "@hemi",
#         ":eri_kernel_h",
#         "@cuda//:cudart_static",
#         "eri_h",
#     ],
# )

# pybind_extension(
#     name = "eri_kernel",
#     srcs = [
#         "eri_kernel.cc",
#     ],
#     copts = [
#         "--std=c++17",
#     ],
#     deps = [
#         ":eri_kernel_cu",
#         "//d4ft/native/xla:custom_call_h",
#     ],
# )

py_test(
    name = "eri_kernel_test",
    srcs = ["eri_kernel_test.py"],
    data = [
        "//d4ft/native/obara_saika:eri_kernel.so",
        "@cuda//:bin",
    ],
    deps = [
        "//d4ft/integral/gto",
        "//d4ft/system",
        "//d4ft/native/xla:custom_call",
        requirement("absl-py"),
        requirement("jax"),
        requirement("jaxlib"),
         "//d4ft/integral/obara_saika",
    ],
)
