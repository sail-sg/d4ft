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

load("@pip_requirements//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "primitive_test",
    srcs = ["primitive_test.py"],
    data = ["//d4ft/system:xyz_files"],
    deps = [
        "//d4ft/integral/gto:cgto",
        "//d4ft/system:mol",
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "symmetry_test",
    srcs = ["symmetry_test.py"],
    deps = [
        "//d4ft/integral/gto:symmetry",
        requirement("absl-py"),
    ],
)
