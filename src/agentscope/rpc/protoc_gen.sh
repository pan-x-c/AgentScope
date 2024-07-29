#!/bin/bash

script_dir="$(dirname "$(readlink -f "$0")")"

cd "${script_dir}/../../.."

protoc -I ./src/agentscope/rpc --grpc_out=./src/agentscope/rpc --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` src/agentscope/rpc/rpc_agent.proto
protoc -I ./src/agentscope/rpc --cpp_out=./src/agentscope/rpc  src/agentscope/rpc/rpc_agent.proto

cd src/agentscope/rpc

g++ -std=c++17 rpc_agent_server.cpp rpc_agent.grpc.pb.cc rpc_agent.pb.cc -o rpc_agent_server `pkg-config --cflags --libs grpc++ grpc protobuf`