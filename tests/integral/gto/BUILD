load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "param_test",
    srcs = ["param_test.py"],
    deps = [
        "//d4ft/integral/gto:gto_utils",
        "//d4ft/system:mol",
    ],
)

py_library(
    name = "symmetry_test",
    srcs = ["symmetry_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//d4ft/system:mol",
        "//d4ft/integral/gto:gto_utils",
        "//d4ft/integral/gto:symmetry",
    ],
)
