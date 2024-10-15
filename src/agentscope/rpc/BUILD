load("@io_bazel_rules_protobuf//:protobuf.bzl", "proto_library", "cc_proto_library")
load("@com_github_grpc_grpc//:build_defs.bzl", "grpc_cc_library")

# 生成 protobuf 文件
proto_library(
    name = "server_proto",
    srcs = ["rpc_agent.proto"],
)

proto_library(
    name = "worker_proto",
    srcs = ["worker_args.proto"],
)
# proto_library(
#     name = "server_proto",
#     srcs = ["rpc_agent.proto", "worker_args.proto"],
# )

cc_proto_library(
    name = "server_cc_proto",
    deps = [":server_proto"]
)

cc_proto_library(
    name = "worker_cc_proto",
    deps = [":worker_proto"]
)

# 生成 gRPC C++ 文件
grpc_cc_library(
    name = "server_grpc",
    srcs = [":server_proto"],
    deps = [
        "@com_github_grpc_grpc//:grpc",
        "@com_google_protobuf//:protobuf",
    ],
)