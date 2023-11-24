# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@pip_requirements//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "contraction",
    srcs = ["contraction.py"],
    deps = [":cgto"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [],
)

py_library(
    name = "cgto",
    srcs = ["cgto.py"],
    deps = [
        ":utils",
        "//d4ft:constants",
        "//d4ft:types",
        "//d4ft:utils",
        "//d4ft/system:mol",
        requirement("jaxtyping"),
    ],
)

py_library(
    name = "symmetry",
    srcs = ["symmetry.py"],
    deps = [
        "//d4ft:types",
        requirement("jaxtyping"),
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

py_library(
    name = "tensorization",
    srcs = ["tensorization.py"],
    data = [
        "//d4ft/native/obara_saika:eri_kernel.so",
        "@cuda//:bin",
    ],
    deps = [
        ":cgto",
        "//d4ft:types",
        "symmetry",
        "//d4ft/native/xla:custom_call",
    ],
)

py_library(
    name = "gto",
    srcs = ["__init__.py"],
    deps = [
        ":cgto",
        ":tensorization",
    ],
)
