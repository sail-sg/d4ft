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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "specs",
    hdrs = ["specs.h"],
)

cc_library(
    name = "custom_call_h",
    hdrs = ["custom_call.h"],
    deps = [
        ":specs",
        # "@cuda//:cudart_static",
    ],
)

py_library(
    name = "custom_call",
    srcs = [
        "__init__.py",
        "custom_call.py",
        requirement("jax"),
        requirement("jaxlib"),
    ],
)
