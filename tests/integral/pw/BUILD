load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "pw_test",
    srcs = ["pw_test.py"],
    deps = [
        "//d4ft:utils",
        "//d4ft/hamiltonian:ortho",
        "//d4ft/integral/pw:pw",
        "//d4ft/system:crystal",
    ],
)
