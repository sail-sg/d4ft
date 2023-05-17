load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "crystal_test",
    srcs = ["crystal_test.py"],
    data = ["//d4ft/system:xyz_files"],
    deps = [
        "//d4ft:utils",
        "//d4ft/system:crystal",
    ],
)
