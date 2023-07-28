# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "overlap_integral_test",
    srcs = ["overlap_integral_test.py"],
    deps = [
        "//d4ft/integral/obara_saika:overlap_integral",
        "@obsa//:obara_saika",
    ],
)

py_test(
    name = "kinetic_integral_test",
    srcs = ["kinetic_integral_test.py"],
    deps = ["//d4ft/integral/obara_saika:kinetic_integral"],
)

py_test(
    name = "nuclear_attraction_integral_test",
    srcs = ["nuclear_attraction_integral_test.py"],
    deps = ["//d4ft/integral/obara_saika:nuclear_attraction_integral"],
)

py_test(
    name = "electron_repulsion_integral_test",
    srcs = ["electron_repulsion_integral_test.py"],
    deps = ["//d4ft/integral/obara_saika:electron_repulsion_integral"],
)

py_test(
    name = "numerical_test",
    srcs = ["numerical_test.py"],
    deps = [
        "//d4ft/integral/obara_saika",
        "@obsa//:obara_saika",
    ],
)
