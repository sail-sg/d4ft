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
    name = "constants",
    srcs = ["constants.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "nuclear",
    srcs = ["nuclear.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "xc",
    srcs = ["xc.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//d4ft/integral/gto:cgto",
        "//d4ft/integral/quadrature:utils",
        "//d4ft/system:mol",
    ],
)

py_library(
    name = "sgd_solver",
    srcs = ["sgd_solver.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":logger",
        ":optimize",
        ":types",
        ":utils",
        "//d4ft/system:mol",
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "scf_solver",
    srcs = ["scf_solver.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":logger",
        ":optimize",
        ":types",
    ],
)

py_library(
    name = "optimize",
    srcs = ["optimize.py"],
    visibility = ["//visibility:public"],
    deps = [":config"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "pyscf_wrapper",
    srcs = ["pyscf_wrapper.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "logger",
    srcs = ["logger.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "d4ft",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":constants",
        ":logger",
        ":nuclear",
        ":optimize",
        ":pyscf_wrapper",
        ":scf_solver",
        ":sgd_solver",
        ":types",
        ":utils",
        ":xc",
    ],
)
