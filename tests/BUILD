load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "solver_test",
    srcs = ["solver_test.py"],
    deps = [
        "//d4ft:sgd_solver",
        "//d4ft/system:mol",
    ],
)
