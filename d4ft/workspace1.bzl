load("@pybind11_bazel//:python_configure.bzl", "python_configure")
load("@rules_cuda//cuda:dependencies.bzl", "rules_cuda_dependencies")

def workspace():
    """Configure pip requirements."""
    python_configure(
        name = "local_config_python",
        python_version = "3",
    )
    rules_cuda_dependencies()

workspace1 = workspace
