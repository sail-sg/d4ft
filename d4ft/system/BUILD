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

py_library(
    name = "occupation",
    srcs = ["occupation.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

filegroup(
    name = "xyz_files",
    srcs = ["xyz_files/diamond.xyz"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "crystal",
    srcs = ["crystal.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//d4ft:constants",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "geometry",
    srcs = ["geometry.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cccdbd",
        ":fake_fullerene",
        ":utils",
    ],
)

py_library(
    name = "mol",
    srcs = ["mol.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":geometry",
        ":occupation",
        ":utils",
        "//d4ft:types",
    ],
)

py_library(
    name = "cccdbd",
    srcs = ["cccdbd.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "fake_fullerene",
    srcs = ["fake_fullerene.py"],
    visibility = ["//visibility:public"],
    deps = [],
)
