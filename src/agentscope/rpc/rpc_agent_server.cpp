#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "rpc_agent.grpc.pb.h"
#include "rpc_agent.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using google::protobuf::Empty;

class RpcAgentServiceImpl final : public RpcAgent::Service {
    Status is_alive(ServerContext* context, const Empty* request, GeneralResponse* response) override {
        response->set_ok(true);
        response->set_message("Server is alive");
        return Status::OK;
    }

    Status stop(ServerContext* context, const Empty* request, GeneralResponse* response) override {
        response->set_ok(true);
        response->set_message("Server is stopping");
        // 实现停止服务器的逻辑
        return Status::OK;
    }

    Status create_agent(ServerContext* context, const CreateAgentRequest* request, GeneralResponse* response) override {
        response->set_ok(true);
        response->set_message("Agent created");
        // 实现创建代理的逻辑
        return Status::OK;
    }

    Status delete_agent(ServerContext* context, const StringMsg* request, GeneralResponse* response) override {
        response->set_ok(true);
        response->set_message("Agent deleted");
        // 实现删除代理的逻辑
        return Status::OK;
    }

    Status delete_all_agents(ServerContext* context, const Empty* request, GeneralResponse* response) override {
        response->set_ok(true);
        response->set_message("All agents deleted");
        // 实现删除所有代理的逻辑
        return Status::OK;
    }

    Status clone_agent(ServerContext* context, const StringMsg* request, GeneralResponse* response) override {
        response->set_ok(true);
        response->set_message("Agent cloned");
        // 实现克隆代理的逻辑
        return Status::OK;
    }

    Status get_agent_list(ServerContext* context, const Empty* request, GeneralResponse* response) override {
        response->set_ok(true);
        response->set_message("Agent list retrieved");
        // 实现获取代理列表的逻辑
        return Status::OK;
    }

    Status get_server_info(ServerContext* context, const Empty* request, GeneralResponse* response) override {
        response->set_ok(true);
        response->set_message("Server info retrieved");
        // 实现获取服务器信息的逻辑
        return Status::OK;
    }

    Status set_model_configs(ServerContext* context, const StringMsg* request, GeneralResponse* response) override {
        response->set_ok(true);
        response->set_message("Model configs updated");
        // 实现更新模型配置的逻辑
        return Status::OK;
    }

    Status get_agent_memory(ServerContext* context, const StringMsg* request, GeneralResponse* response) override {
        response->set_ok(true);
        response->set_message("Agent memory retrieved");
        // 实现获取代理内存的逻辑
        return Status::OK;
    }

    Status call_agent_func(ServerContext* context, const RpcMsg* request, GeneralResponse* response) override {
        response->set_ok(true);
        response->set_message("Agent function called");
        // 实现调用代理函数的逻辑
        return Status::OK;
    }

    Status update_placeholder(ServerContext* context, const UpdatePlaceholderRequest* request, GeneralResponse* response) override {
        response->set_ok(true);
        response->set_message("Placeholder updated");
        // 实现更新占位符的逻辑
        return Status::OK;
    }

    Status download_file(ServerContext* context, const StringMsg* request, grpc::ServerWriter<ByteMsg>* writer) override {
        // 实现文件下载的逻辑
        ByteMsg byte_msg;

        // while (/*condition to continue sending*/) {
        //     writer->Write(byte_msg);
        // }

        return Status::OK;
    }
};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    RpcAgentServiceImpl service;

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;
    server->Wait();
}

int main(int argc, char** argv) {
    RunServer();
    return 0;
}