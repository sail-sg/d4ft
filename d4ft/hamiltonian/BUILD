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

py_library(
    name = "nuclear",
    srcs = ["nuclear.py"],
    deps = [],
)

py_library(
    name = "ortho",
    srcs = ["ortho.py"],
    deps = [],
)

py_library(
    name = "cgto_intors",
    srcs = ["cgto_intors.py"],
    deps = [
        "//d4ft:utils",
        requirement("pyscf"),
    ],
)

py_library(
    name = "dft_cgto",
    srcs = ["dft_cgto.py"],
    deps = [
        ":cgto_intors",
        ":nuclear",
    ],
)

py_library(
    name = "hamiltonian",
    srcs = ["__init__.py"],
    deps = [
        ":cgto_intors",
        ":dft_cgto",
        ":nuclear",
        ":ortho",
    ],
)
