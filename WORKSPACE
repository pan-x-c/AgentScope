# WORKSPACE
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "build_bazel_rules_swift",
    urls = ["https://github.com/bazelbuild/rules_swift/archive/refs/tags/2.1.1.zip"],
    strip_prefix = "rules_swift-2.1.1",
)

# 引入 gRPC
http_archive(
    name = "com_github_grpc_grpc",
    urls = ["https://github.com/grpc/grpc/archive/refs/tags/v1.66.0.zip"],  # 替换为最新版本
    strip_prefix = "grpc-1.66.0",
)

# 引入 Protocol Buffers
http_archive(
    name = "com_google_protobuf",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/refs/tags/v28.2.zip"],  # 替换为最新版本
    strip_prefix = "protobuf-28.2",
)

# 引入 gRPC 相关的构建设定
# load("@grpc//:build_defs.bzl", "grpc_repositories")
# grpc_repositories()
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

http_archive(
    name = "rules_python",
    urls = ["https://github.com/bazelbuild/rules_python/archive/refs/tags/0.36.0.zip"],
    strip_prefix = "rules_python-0.36.0",
)

http_archive(
    name = "pybind11_bazel",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/refs/tags/v2.12.0.zip"],
    strip_prefix = "pybind11_bazel-2.12.0",
)
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11-BUILD.bazel",
    urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip"],  # 适配你的版本
    strip_prefix = "pybind11-2.13.6",
)
# load("@pybind11_bazel//:python_configure.bzl", "python_configure")
# load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
