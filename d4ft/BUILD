# Copyright 2022 Garena Online Private Limited
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

package(default_visibility = ["//visibility:public"])

py_library(
    name = "ao_int",
    srcs = ["ao_int.py"],
)

py_library(
    name = "ao",
    srcs = ["ao.py"],
    deps = [
        ":ao_int",
        ":functions",
    ],
)

py_library(
    name = "functions",
    srcs = ["functions.py"],
)

py_library(
    name = "geometries",
    srcs = ["geometries.py"],
)

py_library(
    name = "geo_opt",
    srcs = ["geo_opt.py"],
    deps = [":grids"],
)

py_library(
    name = "grids",
    srcs = ["grids.py"],
)

py_library(
    name = "loading",
    srcs = ["loading.py"],
)

py_library(
    name = "molecule",
    srcs = ["molecule.py"],
    deps = [
        ":ao",
        ":mo",
    ],
)

py_library(
    name = "mo",
    srcs = ["mo.py"],
    deps = [":functions"],
)

py_library(
    name = "sampler",
    srcs = ["sampler.py"],
)

py_library(
    name = "scf",
    srcs = ["scf.py"],
    deps = [
        ":energy",
        ":functions",
    ],
)

py_library(
    name = "sgd2",
    srcs = ["sgd2.py"],
)

py_library(
    name = "sgd",
    srcs = ["sgd.py"],
    deps = [
        ":energy",
        ":sampler",
        ":molecule",
    ],
)

py_library(
    name = "sscf",
    srcs = ["sscf.py"],
)

py_library(
    name = "visualization",
    srcs = ["visualization.py"],
)

py_library(
    name = "vpf",
    srcs = ["vpf.py"],
)

py_library(
    name = "energy",
    srcs = ["energy.py"],
)

py_library(
    name = "d4ft",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":ao",
        ":ao_int",
        ":functions",
        ":geo_opt",
        ":geometries",
        ":grids",
        ":loading",
        ":mo",
        ":molecule",
        ":sampler",
        ":scf",
        ":sgd",
        ":sgd2",
        ":sscf",
        ":visualization",
        ":vpf",
    ],
)
