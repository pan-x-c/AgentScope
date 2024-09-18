# -*- coding: utf-8 -*-
"""A proxy object which represent a object located in a rpc server."""
from typing import Any, Callable
from abc import ABC
from inspect import getmembers, isfunction
from types import FunctionType
from concurrent.futures import ThreadPoolExecutor, Future
import threading

try:
    import cloudpickle as pickle
except ImportError as e:
    from agentscope.utils.common import ImportErrorReporter

    pickle = ImportErrorReporter(e, "distribute")

from .rpc_client import RpcClient
from .rpc_async import AsyncResult
from ..exception import AgentCreationError, AgentServerNotAliveError


def get_public_methods(cls: type) -> list[str]:
    """Get all public methods of the given class."""
    return [
        name
        for name, member in getmembers(cls, predicate=isfunction)
        if isinstance(member, FunctionType) and not name.startswith("_")
    ]


def _call_func_in_thread(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call a function in a sub-thread."""
    future = Future()

    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            result = func(*args, **kwargs)
            future.set_result(result)
        except Exception as ex:
            future.set_exception(ex)

    thread = threading.Thread(target=wrapper, args=args, kwargs=kwargs)
    thread.start()

    return future


class RpcObject(ABC):
    """A proxy object which represent an object located in a rpc server."""

    def __init__(
        self,
        cls: type,
        oid: str,
        host: str,
        port: int,
        connect_existing: bool = False,
        max_pool_size: int = 8192,
        max_expire_time: int = 7200,
        max_timeout_seconds: int = 5,
        local_mode: bool = True,
        configs: dict = None,
    ) -> None:
        """Initialize the rpc object.

        Args:
            cls (`type`): The class of the object in the rpc server.
            oid (`str`): The id of the object in the rpc server.
            host (`str`): The host of the rpc server.
            port (`int`): The port of the rpc server.
            max_pool_size (`int`, defaults to `8192`):
                Max number of task results that the server can accommodate.
            max_expire_time (`int`, defaults to `7200`):
                Max expire time for task results.
            max_timeout_seconds (`int`, defaults to `5`):
                Max timeout seconds for the rpc call.
            local_mode (`bool`, defaults to `True`):
                Whether the started gRPC server only listens to local
                requests.
            connect_existing (`bool`, defaults to `False`):
                Set to `True`, if the object is already running on the
                server.
        """
        self.host = host
        self.port = port
        self._oid = oid
        self._cls = cls
        self.connect_existing = connect_existing
        self.executor = ThreadPoolExecutor(max_workers=1)

        from ..studio._client import _studio_client

        if self.port is None and _studio_client.active:
            server = _studio_client.alloc_server()
            if "host" in server:
                if RpcClient(
                    host=server["host"],
                    port=server["port"],
                ).is_alive():
                    self.host = server["host"]
                    self.port = server["port"]
        launch_server = self.port is None
        self.server_launcher = None
        if launch_server:
            from ..server import RpcAgentServerLauncher

            # check studio first
            self.host = "localhost"
            studio_url = None
            if _studio_client.active:
                studio_url = _studio_client.studio_url
            self.server_launcher = RpcAgentServerLauncher(
                host=self.host,
                port=self.port,
                capacity=2,
                max_pool_size=max_pool_size,
                max_expire_time=max_expire_time,
                max_timeout_seconds=max_timeout_seconds,
                local_mode=local_mode,
                custom_agent_classes=[cls],
                studio_url=studio_url,  # type: ignore[arg-type]
            )
            self._launch_server()
        else:
            self.client = RpcClient(self.host, self.port)
        if not connect_existing:
            self.create(configs)
            if launch_server:
                self._check_created()
        else:
            self._creating_stub = None

    def create(self, configs: dict) -> None:
        """create the object on the rpc server."""
        self._creating_stub = AsyncResult(
            host=self.host,
            port=self.port,
            task_id=self.client._create_agent_async(  # pylint: disable=W0212
                configs,
                self._oid,
            ),
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._check_created()
        if "__call__" in self._cls._async_func:
            return self._async_func("__call__")(*args, **kwargs)
        else:
            return self._call_func(
                "__call__",
                args={
                    "args": args,
                    "kwargs": kwargs,
                },
            )

    def __getitem__(self, item: str) -> Any:
        return self._call_func("__getitem__", {"args": (item,)})

    def _launch_server(self) -> None:
        """Launch a rpc server and update the port and the client"""
        self.server_launcher.launch()
        self.port = self.server_launcher.port
        self.client = RpcClient(
            host=self.host,
            port=self.port,
        )
        if not self.client.is_alive():
            raise AgentServerNotAliveError(self.host, self.port)

    def stop(self) -> None:
        """Stop the RpcAgent and the rpc server."""
        if self.server_launcher is not None:
            self.server_launcher.shutdown()

    def _check_created(self) -> None:
        """Check if the object is created on the rpc server."""
        if self._creating_stub is not None:
            try:
                response = self._creating_stub.result()
            except Exception as ex:
                raise AgentCreationError(self.host, self.port) from ex
            if response is not True:
                raise AgentCreationError(self.host, self.port)
            self._creating_stub = None

    def _call_func(self, func_name: str, args: dict) -> Any:
        """Call a function in rpc server."""
        return pickle.loads(
            self.client.call_agent_func(
                agent_id=self._oid,
                func_name=func_name,
                value=pickle.dumps(args),
            ),
        )

    def _async_func(self, name: str) -> Callable:
        def async_wrapper(*args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
            return AsyncResult(
                host=self.host,
                port=self.port,
                stub=_call_func_in_thread(
                    self._call_func,
                    func_name=name,
                    args={"args": args, "kwargs": kwargs},
                ),
            )

        return async_wrapper

    def _sync_func(self, name: str) -> Callable:
        def sync_wrapper(*args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
            return self._call_func(
                func_name=name,
                args={"args": args, "kwargs": kwargs},
            )

        return sync_wrapper

    def __getattr__(self, name: str) -> Callable:
        self._check_created()
        if name in self._cls._async_func:
            # for async functions
            return self._async_func(name)

        elif name in self._cls._sync_func:
            # for sync functions
            return self._sync_func(name)

        else:
            # for attributes
            return self._call_func(
                func_name=name,
                args={},
            )

    def __del__(self) -> None:
        self.stop()

    def __deepcopy__(self, memo: dict) -> Any:
        """For deepcopy."""
        if id(self) in memo:
            return memo[id(self)]

        clone = RpcObject(
            cls=self._cls,
            oid=self._oid,
            host=self.host,
            port=self.port,
            connect_existing=True,
        )
        memo[id(self)] = clone

        return clone

    def __reduce__(self) -> tuple:
        self._check_created()
        return (
            RpcObject,
            (
                self._cls,
                self._oid,
                self.host,
                self.port,
                True,
            ),
        )
