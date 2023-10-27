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

filegroup(
    name = "xyz_files",
    srcs = glob(["xyz_files/*.xyz"]),
)

py_library(
    name = "occupation",
    srcs = ["occupation.py"],
    deps = [],
)

py_library(
    name = "crystal",
    srcs = ["crystal.py"],
    deps = [
        "//d4ft:constants",
        "//d4ft:types",
        requirement("ase"),
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [],
)

py_library(
    name = "geometry",
    srcs = ["geometry.py"],
    deps = [
        ":cccdbd",
        ":fake_fullerene",
        ":refdata",
        ":utils",
        requirement("pubchempy"),
    ],
)

py_library(
    name = "mol",
    srcs = ["mol.py"],
    deps = [
        ":geometry",
        ":occupation",
        ":utils",
        "//d4ft:types",
        requirement("pyscf"),
    ],
)

py_library(
    name = "cccdbd",
    srcs = ["cccdbd.py"],
    deps = [
        requirement("bs4"),
    ],
)

py_library(
    name = "refdata",
    srcs = ["refdata.py"],
    deps = [
        requirement("pandas"),
    ],
)

py_library(
    name = "fake_fullerene",
    srcs = ["fake_fullerene.py"],
    deps = [],
)

py_library(
    name = "system",
    srcs = ["__init__.py"],
    data = ["//d4ft/system:xyz_files"],
    deps = [
        ":cccdbd",
        ":crystal",
        ":fake_fullerene",
        ":geometry",
        ":mol",
        ":occupation",
        ":utils",
    ],
)
