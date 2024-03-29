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
    name = "drivers",
    srcs = ["drivers.py"],
    deps = [
        "//d4ft:config",
        "//d4ft:logger",
        "//d4ft:optimize",
        "//d4ft:types",
        "//d4ft:utils",
        "//d4ft:xc",
        "//d4ft/hamiltonian:cgto_intors",
        "//d4ft/hamiltonian:mf_cgto",
        "//d4ft/hamiltonian:ortho",
        "//d4ft/integral/gto:cgto",
        "//d4ft/integral/obara_saika:driver",
        "//d4ft/integral/quadrature:grids",
        "//d4ft/solver:pyscf_wrapper",
        "//d4ft/solver:scf",
        "//d4ft/solver:sgd",
        "//d4ft/system:mol",
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
        requirement("jax-xc"),
        requirement("jaxtyping"),
    ],
)

py_library(
    name = "sgd",
    srcs = ["sgd.py"],
    deps = [
        "//d4ft:config",
        "//d4ft:logger",
        "//d4ft:optimize",
        "//d4ft:types",
        "//d4ft:utils",
        "//d4ft/system:mol",
        requirement("chex"),
        requirement("optax"),
    ],
)

py_library(
    name = "scf",
    srcs = ["scf.py"],
    deps = [
        "//d4ft:config",
        "//d4ft:logger",
        "//d4ft:optimize",
        "//d4ft:types",
        "//d4ft/integral/gto:cgto",
    ],
)

py_library(
    name = "pyscf_wrapper",
    srcs = ["pyscf_wrapper.py"],
    deps = [],
)

py_library(
    name = "solver",
    srcs = ["__init__.py"],
    deps = [
        ":drivers",
        ":pyscf_wrapper",
        ":scf",
        ":sgd",
    ],
)
