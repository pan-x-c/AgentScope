# -*- coding: utf-8 -*-
"""Related functions for cpp server."""

from loguru import logger

try:
    import cloudpickle as pickle
except ImportError as import_error:
    from agentscope.utils.common import ImportErrorReporter

    pickle = ImportErrorReporter(import_error, "distribute")

from agentscope.rpc.rpc_object import RpcObject
from agentscope.rpc.rpc_meta import RpcMeta


def create_agent(agent_id: str, agent_init_args: str, host: str, port: int):
    agent_configs = pickle.loads(agent_init_args)
    cls_name = agent_configs["class_name"]
    try:
        cls = RpcMeta.get_class(cls_name)
    except ValueError as e:
        err_msg = (f"Class [{cls_name}] not found: {str(e)}",)
        logger.error(err_msg)
        return None, str(err_msg)
    try:
        instance = cls(
            *agent_configs["args"],
            **agent_configs["kwargs"],
        )
    except Exception as e:
        err_msg = f"Failed to create agent instance <{cls_name}>: {str(e)}"

        logger.error(err_msg)
        return None, err_msg

    # Reset the __reduce_ex__ method of the instance
    # With this method, all objects stored in agent_pool will be serialized
    # into their Rpc version
    rpc_init_cfg = (
        cls,
        agent_id,
        host,
        port,
        True,
    )
    instance._dist_config = {  # pylint: disable=W0212
        "args": rpc_init_cfg,
    }

    def to_rpc(obj, _) -> tuple:  # type: ignore[no-untyped-def]
        return (
            RpcObject,
            obj._dist_config["args"],  # pylint: disable=W0212
        )

    instance.__reduce_ex__ = to_rpc.__get__(  # pylint: disable=E1120
        instance,
    )
    instance._oid = agent_id  # pylint: disable=W0212
    logger.info(f"create agent instance <{cls_name}>[{agent_id}] [{instance.name}]")
    return instance, ""
