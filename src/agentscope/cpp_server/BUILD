load("@pybind11_bazel//:build_defs.bzl", "pybind_library")
# load("@pybind11//:workspace.bzl", "pybind11_library")

# pybind11_library(
pybind_library(
    name = "cpp_server",
    srcs = [
        "worker.cc",
        "rpc_agent_servicer.cc",
    ],
    hdrs = ["worker.h"],
    deps = [
        "//src/agentscope/rpc:worker_cc_proto",
        "//src/agentscope/rpc:server_grpc",
        # "@pybind11//:pybind11",
        # "@rules_python//python/cc:current_py_cc_libs",
        "@com_github_grpc_grpc//:grpc++"
    ],
)