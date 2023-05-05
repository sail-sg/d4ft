load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "primitive_test",
    srcs = ["primitive_test.py"],
    deps = [
        "//d4ft/integral/gto:cgto",
        "//d4ft/system:mol",
    ],
)

py_library(
    name = "symmetry_test",
    srcs = ["symmetry_test.py"],
    deps = ["//d4ft/integral/gto:symmetry"],
)
