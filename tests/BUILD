load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "ao_test",
    srcs = ["ao_test.py"],
    deps = ["//d4ft:ao"],
)

py_test(
    name = "molecule_test",
    srcs = ["molecule_test.py"],
    deps = [
        "//d4ft:geometries",
        "//d4ft:molecule",
        "//d4ft:sgd",
    ],
)
