load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "example_h",
    hdrs = ["example.h"],
    copts = [
        "--std=c++17",
    ],
    deps = [
        "//d4ft/native/xla:specs",
    ],
)

cuda_library(
    name = "example_cu",
    srcs = ["example.cu"],
    copts = [
        "--std=c++17",
    ],
    deps = [
        ":example_h",
    ],
)

pybind_extension(
    name = "example",
    srcs = [
        "example.cc",
    ],
    copts = [
        "--std=c++17",
    ],
    deps = [
        ":example_cu",
        "//d4ft/native/xla:custom_call_h",
    ],
)

py_test(
    name = "example_test",
    srcs = ["example_test.py"],
    data = [":example.so"],
    deps = [
        "//d4ft/native/xla:custom_call",
        requirement("jax"),
        requirement("jaxlib"),
    ],
)
