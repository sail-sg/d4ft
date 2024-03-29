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
    name = "driver",
    srcs = ["driver.py"],
    deps = [
        ":angular_stats",
        ":electron_repulsion_integral",
        ":kinetic_integral",
        ":nuclear_attraction_integral",
        "//d4ft/integral/gto:cgto",
        "//d4ft/integral/gto:symmetry",
        "//d4ft/integral/gto:tensorization",
        "//d4ft/integral/obara_saika",
        requirement("tqdm"),
    ],
)

py_library(
    name = "boys_table",
    srcs = ["boys_table.py"],
)

py_library(
    name = "boys",
    srcs = ["boys.py"],
    deps = [":boys_table"],
)

py_library(
    name = "terms",
    srcs = ["terms.py"],
    deps = [],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [],
)

py_library(
    name = "angular_stats",
    srcs = ["angular_stats.py"],
    deps = ["//d4ft:types"],
)

py_library(
    name = "overlap_integral",
    srcs = ["overlap_integral.py"],
    deps = [
        ":angular_stats",
        ":terms",
        ":utils",
        "//d4ft:types",
        "//d4ft/integral/gto:cgto",
    ],
)

py_library(
    name = "kinetic_integral",
    srcs = ["kinetic_integral.py"],
    deps = [
        ":angular_stats",
        ":terms",
        "//d4ft:types",
        "//d4ft/integral/gto:cgto",
    ],
)

py_library(
    name = "nuclear_attraction_integral",
    srcs = ["nuclear_attraction_integral.py"],
    deps = [
        ":angular_stats",
        ":boys",
        ":terms",
        ":utils",
        "//d4ft:types",
        "//d4ft/integral/gto:cgto",
    ],
)

py_library(
    name = "electron_repulsion_integral",
    srcs = ["electron_repulsion_integral.py"],
    deps = [
        ":angular_stats",
        ":boys",
        ":terms",
        ":utils",
        "//d4ft:types",
        "//d4ft/integral/gto:cgto",
    ],
)

py_library(
    name = "obara_saika",
    srcs = ["__init__.py"],
    deps = [
        ":angular_stats",
        ":boys",
        ":boys_table",
        ":electron_repulsion_integral",
        ":kinetic_integral",
        ":nuclear_attraction_integral",
        ":overlap_integral",
        ":terms",
        ":utils",
    ],
)
