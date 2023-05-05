load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "solver_test",
    srcs = ["solver_test.py"],
    deps = [
        "//d4ft:nuclear",
        "//d4ft:sgd_solver",
        "//d4ft:xc",
        "//d4ft/hamiltonian:cgto_intors",
        "//d4ft/hamiltonian:dft_cgto",
        "//d4ft/hamiltonian:ortho",
        "//d4ft/integral/gto:cgto",
        "//d4ft/integral/obara_saika:driver",
        "//d4ft/system:mol",
    ],
)
