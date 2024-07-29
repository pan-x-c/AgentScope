from agentscope.rpc.rpc_agent_client import RpcAgentClient

client = RpcAgentClient(host="localhost", port=50051)
print(client.is_alive())
