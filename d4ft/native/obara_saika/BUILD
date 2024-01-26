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
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "comb",
    hdrs = ["comb.h"],
)

cc_library(
    name = "boys",
    hdrs = ["boys.h"],
    deps = [
        "//d4ft/native/gamma:igamma",
        "//d4ft/native/gamma:lgamma",
        "@hemi",
        ":comb",
    ],
)

cc_library(
    name = "eri",
    hdrs = ["eri.h"],
    deps = [
        ":boys",
        ":comb",
        "@hemi",
    ],
)


cc_library(
    name = "eri_kernel_h",
    hdrs = ["eri_kernel.h"],
    copts = [
        "--std=c++17",
    ],
    deps = [
        "@hemi",
        "//d4ft/native/xla:specs",
    ],
)

cuda_library(
    name = "eri_kernel_cu",
    srcs = ["eri_kernel.cu"],
    hdrs = ["eri_kernel.h"],
    copts = [
        "--expt-extended-lambda",
        "--expt-relaxed-constexpr",
    ],
    linkopts = [
        "-lstdc++",
    ],
    deps = [
        ":eri",
        "@hemi",
        "//d4ft/native/xla:specs",
    ],
)

# cc_binary(
#     name = "eri_test",
#     srcs = ["eri_test.cc"],
#     deps = [
#         ":eri_kernel",
#         "@cuda//:cudart_static",
#         "@hemi",
#     ],
# )

cc_binary(
    name = "boys_test",
    srcs = ["boys_test.cc"],
    deps = [
        ":boys",
        "@cuda//:cudart_static",
    ],
)

pybind_extension(
    name = "eri_kernel",
    srcs = [
        "eri_kernel.cc",
    ],
    copts = [
        "--std=c++17",
    ],
    deps = [
        ":eri_kernel_cu",
        "//d4ft/native/xla:custom_call_h",
    ],
)