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

load("@rules_python//python:defs.bzl", "py_library")
load("@pip_requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

exports_files([
    "workspace0.bzl",
    "workspace1.bzl",
    "workspace2.bzl",
])

py_library(
    name = "constants",
    srcs = ["constants.py"],
    deps = [],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        requirement("dm-haiku"),
        requirement("optax"),
        requirement("jaxtyping"),
    ],
)

py_library(
    name = "xc",
    srcs = ["xc.py"],
    deps = [
        ":types",
        "//d4ft/integral/gto:cgto",
        "//d4ft/integral/quadrature:utils",
        "//d4ft/system:mol",
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        requirement("ml_collections"),
        requirement("pydantic"),
    ],
)

py_library(
    name = "optimize",
    srcs = ["optimize.py"],
    deps = [
        ":config",
        ":types",
        requirement("optax"),
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        requirement("einops"),
        requirement("ml_collections"),
    ],
)

py_library(
    name = "logger",
    srcs = ["logger.py"],
    deps = [
        requirement("pandas"),
    ],
)

py_library(
    name = "d4ft",
    srcs = ["__init__.py"],
    deps = [
        ":config",
        ":constants",
        ":logger",
        ":optimize",
        ":types",
        ":utils",
        ":xc",
        "//d4ft/hamiltonian",
        "//d4ft/integral",
        "//d4ft/native/xla:custom_call",
        "//d4ft/solver",
        "//d4ft/system",
    ],
)
