package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "lgamma",
    hdrs = ["lgamma.h"],
    deps = [":constants"],
)

cc_library(
    name = "digamma",
    hdrs = ["digamma.h"],
    deps = [":constants"],
)

cc_library(
    name = "igamma",
    hdrs = ["igamma.h"],
    deps = [
        ":constants",
        ":digamma",
        ":lgamma",
    ],
)
