# -*- coding: utf-8 -*-
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
try:
    import grpc
    from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
except ImportError as import_error:
    from agentscope.utils.tools import ImportErrorReporter

    grpc = ImportErrorReporter(import_error, "distribute")
    google_dot_protobuf_dot_empty__pb2 = ImportErrorReporter(
        import_error,
        "distribute",
    )
import agentscope.rpc.rpc_agent_pb2 as rpc__agent__pb2


class RpcAgentStub(object):
    """Servicer for rpc agent server"""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.is_alive = channel.unary_unary(
            "/RpcAgent/is_alive",
            request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            response_deserializer=rpc__agent__pb2.StatusResponse.FromString,
        )
        self.call_func = channel.unary_unary(
            "/RpcAgent/call_func",
            request_serializer=rpc__agent__pb2.RpcMsg.SerializeToString,
            response_deserializer=rpc__agent__pb2.RpcMsg.FromString,
        )


class RpcAgentServicer(object):
    """Servicer for rpc agent server"""

    def is_alive(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def call_func(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_RpcAgentServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "is_alive": grpc.unary_unary_rpc_method_handler(
            servicer.is_alive,
            request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            response_serializer=rpc__agent__pb2.StatusResponse.SerializeToString,
        ),
        "call_func": grpc.unary_unary_rpc_method_handler(
            servicer.call_func,
            request_deserializer=rpc__agent__pb2.RpcMsg.FromString,
            response_serializer=rpc__agent__pb2.RpcMsg.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "RpcAgent",
        rpc_method_handlers,
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class RpcAgent(object):
    """Servicer for rpc agent server"""

    @staticmethod
    def is_alive(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/is_alive",
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            rpc__agent__pb2.StatusResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def call_func(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/call_func",
            rpc__agent__pb2.RpcMsg.SerializeToString,
            rpc__agent__pb2.RpcMsg.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
