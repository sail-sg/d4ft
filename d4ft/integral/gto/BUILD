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

py_library(
    name = "contraction",
    srcs = ["contraction.py"],
    visibility = ["//visibility:public"],
    deps = [":cgto"],
)

py_library(
    name = "cgto",
    srcs = ["cgto.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//d4ft:constants",
        "//d4ft:utils",
        "//d4ft/system:mol",
    ],
)

py_library(
    name = "symmetry",
    srcs = ["symmetry.py"],
    visibility = ["//visibility:public"],
    deps = ["//d4ft:types"],
)

py_library(
    name = "tensorization",
    srcs = ["tensorization.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cgto",
        "//d4ft:types",
    ],
)
