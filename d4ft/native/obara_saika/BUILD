load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "comb",
    hdrs = ["comb.h"],
)

cc_library(
    name = "boys",
    hdrs = ["boys.h"],
    deps = [
        "//d4ft/native/gamma:igamma",
        "//d4ft/native/gamma:lgamma",
    ],
)

cc_library(
    name = "eri",
    hdrs = ["eri.h"],
    deps = [
        ":boys",
        ":comb",
    ],
)

cuda_library(
    name = "eri_kernel",
    srcs = ["eri_kernel.cu"],
    hdrs = ["eri_kernel.h"],
    deps = [
        ":eri",
        "@hemi",
    ],
)

cc_binary(
    name = "eri_test",
    srcs = ["eri_test.cc"],
    deps = [
        ":eri_kernel",
        "@hemi",
    ],
)

cc_binary(
    name = "boys_test",
    srcs = ["boys_test.cc"],
    deps = [
        ":boys",
    ],
)
