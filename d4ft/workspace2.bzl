load("@rules_cc//cc:repositories.bzl", "rules_cc_toolchains")

def workspace():
    rules_cc_toolchains()

workspace2 = workspace
