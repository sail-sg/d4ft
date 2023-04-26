load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

http_archive(
    name = "rules_python",
    sha256 = "8c8fe44ef0a9afc256d1e75ad5f448bb59b81aba149b8958f02f7b3a98f5d9b4",
    strip_prefix = "rules_python-0.13.0",
    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.13.0.tar.gz",
)

new_git_repository(
    name = "obsa",
    branch = "master",
    build_file = "//third_party:obsa.BUILD",
    remote = "https://github.com/bast/obara-saika.git",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_cuda",
    sha256 = "f80438bee9906e9ecb1a8a4ae2365374ac1e8a283897281a2db2fb7fcf746333",
    strip_prefix = "runtime-b1c7cce21ba4661c17ac72421c6a0e2015e7bef3/third_party/rules_cuda",
    urls = ["https://github.com/tensorflow/runtime/archive/b1c7cce21ba4661c17ac72421c6a0e2015e7bef3.tar.gz"],
)

load("@rules_cuda//cuda:dependencies.bzl", "rules_cuda_dependencies")

rules_cuda_dependencies()

load("@rules_cc//cc:repositories.bzl", "rules_cc_toolchains")

rules_cc_toolchains()
