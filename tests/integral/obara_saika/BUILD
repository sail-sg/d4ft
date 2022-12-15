load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "overlap_integral_test",
    srcs = ["overlap_integral_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//d4ft/integral/obara_saika:overlap_integral",
        "@obsa//:obara_saika",
    ],
)

py_test(
    name = "kinetic_integral_test",
    srcs = ["kinetic_integral_test.py"],
    visibility = ["//visibility:public"],
    deps = ["//d4ft/integral/obara_saika:kinetic_integral"],
)

py_test(
    name = "nuclear_attraction_integral_test",
    srcs = ["nuclear_attraction_integral_test.py"],
    visibility = ["//visibility:public"],
    deps = ["//d4ft/integral/obara_saika:nuclear_attraction_integral"],
)

py_test(
    name = "electron_repulsion_integral_test",
    srcs = ["electron_repulsion_integral_test.py"],
    visibility = ["//visibility:public"],
    deps = ["//d4ft/integral/obara_saika:electron_repulsion_integral"],
)

py_test(
    name = "tensorize_test",
    srcs = ["tensorize_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//d4ft/integral/obara_saika:electron_repulsion_integral",
        "//d4ft/integral/obara_saika:kinetic_integral",
        "//d4ft/integral/obara_saika:nuclear_attraction_integral",
        "//d4ft/integral/obara_saika:overlap_integral",
        "//d4ft/integral/obara_saika:utils",
    ],
)

py_test(
    name = "numerical_test",
    srcs = ["numerical_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//d4ft/integral/obara_saika",
        "@obsa//:obara_saika",
    ],
)
