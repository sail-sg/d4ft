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

"""D4FT workspace dependencies, loaded in WORKSPACE."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def workspace():
    """Load requested packages."""

    # Add bazel_skylib dependency first
    maybe(
        http_archive,
        name = "bazel_skylib",
        sha256 = "66ffd9315665bfaafc96b52278f57c7e2dd09f5ede279ea6d39b2be471e7e3aa",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
        ],
    )

    # Add rules_cc dependency
    maybe(
        http_archive,
        name = "rules_cc",
        sha256 = "2037875b9a4456dce4a79d112a8ae885bbc4aad968e6587dca6e64f3a0900cdf",
        strip_prefix = "rules_cc-0.0.9",
        urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.9/rules_cc-0.0.9.tar.gz"],
    )

    # # this version doesn't work
    # maybe(
    #     http_archive,
    #     name = "rules_python",
    #     sha256 = "8c8fe44ef0a9afc256d1e75ad5f448bb59b81aba149b8958f02f7b3a98f5d9b4",
    #     strip_prefix = "rules_python-0.13.0",
    #     url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.13.0.tar.gz",
    # )

    maybe(
        http_archive,
        name = "rules_python",
        sha256 = "b593d13bb43c94ce94b483c2858e53a9b811f6f10e1e0eedc61073bd90e58d9c",
        strip_prefix = "rules_python-0.12.0",
        urls = [
            "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.12.0.tar.gz",
        ],
    )

    maybe(
        new_git_repository,
        name = "obsa",
        branch = "master",
        build_file = "//third_party:obsa.BUILD",
        remote = "https://github.com/bast/obara-saika.git",
    )

    maybe(
        http_archive,
        name = "rules_cuda",
        sha256 = "c92b334d769a07cd991b7675b2f6076b8b95cd3b28b14268a2f379f8baae58e0",
        strip_prefix = "rules_cuda-v0.2.3",
        urls = ["https://github.com/bazel-contrib/rules_cuda/releases/download/v0.2.3/rules_cuda-v0.2.3.tar.gz"],
    )

    maybe(
        new_git_repository,
        name = "hemi",
        branch = "master",
        build_file = "//third_party:hemi.BUILD",
        recursive_init_submodules = True,
        remote = "https://github.com/harrism/hemi.git",
    )

    pybind11_bazel_version = "fc56ce8a8b51e3dd941139d329b63ccfea1d304b"  # Latest @ 2023-04-27
    maybe(
        http_archive,
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-{}".format(pybind11_bazel_version),
        urls = ["https://github.com/pybind/pybind11_bazel/archive/{}.zip".format(pybind11_bazel_version)],
    )

    # We still require the pybind library.
    pybind11_version = "2.10.4"
    maybe(
        http_archive,
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        strip_prefix = "pybind11-{}".format(pybind11_version),
        urls = ["https://github.com/pybind/pybind11/archive/v{}.tar.gz".format(pybind11_version)],
    )


    maybe(
        http_archive,
        name = "bazel_clang_tidy",
        sha256 = "ec8c5bf0c02503b928c2e42edbd15f75e306a05b2cae1f34a7bc84724070b98b",
        strip_prefix = "bazel_clang_tidy-783aa523aafb4a6798a538c61e700b6ed27975a7",
        urls = [
            "https://github.com/erenon/bazel_clang_tidy/archive/783aa523aafb4a6798a538c61e700b6ed27975a7.zip",
        ],
    )

workspace0 = workspace
