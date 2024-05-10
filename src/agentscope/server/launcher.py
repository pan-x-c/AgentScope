# -*- coding: utf-8 -*-
""" Server of distributed agent"""
from multiprocessing import Process, Event, Pipe
from multiprocessing.synchronize import Event as EventClass
import socket
import asyncio
import signal
import argparse
from typing import Any, Type, Optional
from concurrent import futures
from loguru import logger

try:
    import grpc
except ImportError:
    grpc = None

from .servicer import AgentPlatform
from ..agents.agent import AgentBase
from ..utils.tools import _get_timestamp

try:
    from ..rpc.rpc_agent_pb2_grpc import (
        add_RpcAgentServicer_to_server,
    )
except ModuleNotFoundError:
    add_RpcAgentServicer_to_server = Any


def setup_agent_server(
    host: str,
    port: int,
    server_id: str,
    init_settings: dict = None,
    start_event: EventClass = None,
    stop_event: EventClass = None,
    pipe: int = None,
    local_mode: bool = True,
    max_pool_size: int = 8192,
    max_timeout_seconds: int = 1800,
    custom_agents: list = None,
) -> None:
    """Setup agent server.

    Args:
        host (`str`, defaults to `"localhost"`):
            Hostname of the agent server.
        port (`int`):
            The socket port monitored by the agent server.
        server_id (`str`):
            The id of the server.
        init_settings (`dict`, defaults to `None`):
            Init settings for agentscope.init.
        start_event (`EventClass`, defaults to `None`):
            An Event instance used to determine whether the child process
            has been started.
        stop_event (`EventClass`, defaults to `None`):
            The stop Event instance used to determine whether the child
            process has been stopped.
        pipe (`int`, defaults to `None`):
            A pipe instance used to pass the actual port of the server.
        local_mode (`bool`, defaults to `None`):
            Only listen to local requests.
        max_pool_size (`int`, defaults to `8192`):
            Max number of task results that the server can accommodate.
        max_timeout_seconds (`int`, defaults to `1800`):
            Timeout for task results.
        custom_agents (`list`, defaults to `None`):
            A list of custom agent classes that are not in `agentscope.agents`.
    """
    asyncio.run(
        setup_agent_server_async(
            host=host,
            port=port,
            server_id=server_id,
            init_settings=init_settings,
            start_event=start_event,
            stop_event=stop_event,
            pipe=pipe,
            local_mode=local_mode,
            max_pool_size=max_pool_size,
            max_timeout_seconds=max_timeout_seconds,
            custom_agents=custom_agents,
        ),
    )


async def setup_agent_server_async(
    host: str,
    port: int,
    server_id: str,
    init_settings: dict = None,
    start_event: EventClass = None,
    stop_event: EventClass = None,
    pipe: int = None,
    local_mode: bool = True,
    max_pool_size: int = 8192,
    max_timeout_seconds: int = 1800,
    custom_agents: list = None,
) -> None:
    """Setup agent server in an async way.

    Args:
        host (`str`, defaults to `"localhost"`):
            Hostname of the agent server.
        port (`int`):
            The socket port monitored by the agent server.
        server_id (`str`):
            The id of the server.
        init_settings (`dict`, defaults to `None`):
            Init settings for agentscope.init.
        start_event (`EventClass`, defaults to `None`):
            An Event instance used to determine whether the child process
            has been started.
        stop_event (`EventClass`, defaults to `None`):
            The stop Event instance used to determine whether the child
            process has been stopped.
        pipe (`int`, defaults to `None`):
            A pipe instance used to pass the actual port of the server.
        local_mode (`bool`, defaults to `None`):
            Only listen to local requests.
        max_pool_size (`int`, defaults to `8192`):
            Max number of task results that the server can accommodate.
        max_timeout_seconds (`int`, defaults to `1800`):
            Timeout for task results.
        custom_agents (`list`, defaults to `None`):
            A list of custom agent classes that are not in `agentscope.agents`.
    """
    from agentscope._init import init_process

    if init_settings is not None:
        init_process(**init_settings)
    servicer = AgentPlatform(
        host=host,
        port=port,
        max_pool_size=max_pool_size,
        max_timeout_seconds=max_timeout_seconds,
    )
    # update agent registry
    if custom_agents is not None:
        for agent_class in custom_agents:
            AgentBase.register_agent_class(agent_class=agent_class)

    async def shutdown_signal_handler() -> None:
        logger.info(
            f"Received shutdown signal. Gracefully stopping the server at "
            f"[{host}:{port}].",
        )
        await server.stop(grace=5)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(shutdown_signal_handler()),
        )
    while True:
        try:
            port = check_port(port)
            servicer.port = port
            server = grpc.aio.server(
                futures.ThreadPoolExecutor(max_workers=None),
            )
            add_RpcAgentServicer_to_server(servicer, server)
            if local_mode:
                server.add_insecure_port(f"localhost:{port}")
            else:
                server.add_insecure_port(f"0.0.0.0:{port}")
            await server.start()
            break
        except OSError:
            logger.warning(
                f"Failed to start agent server at port [{port}]"
                f"try another port",
            )
    logger.info(
        f"agent server [{server_id}] at {host}:{port} started successfully",
    )
    if start_event is not None:
        pipe.send(port)
        start_event.set()
        while not stop_event.is_set():
            await asyncio.sleep(1)
        logger.info(
            f"Stopping agent server at [{host}:{port}]",
        )
        await server.stop(10.0)
    else:
        await server.wait_for_termination()
    logger.info(
        f"agent server [{server_id}] at {host}:{port} stopped successfully",
    )


def find_available_port() -> int:
    """Get an unoccupied socket port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def check_port(port: Optional[int] = None) -> int:
    """Check if the port is available.

    Args:
        port (`int`):
            the port number being checked.

    Returns:
        `int`: the port number that passed the check. If the port is found
        to be occupied, an available port number will be automatically
        returned.
    """
    if port is None:
        new_port = find_available_port()
        logger.warning(
            "agent server port is not provided, automatically select "
            f"[{new_port}] as the port number.",
        )
        return new_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", port)) == 0:
            new_port = find_available_port()
            logger.warning(
                f"Port [{port}] is occupied, use [{new_port}] instead",
            )
            return new_port
    return port


class AgentServerLauncher:
    """The launcher of AgentPlatform (formerly RpcAgentServer)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = None,
        max_pool_size: int = 8192,
        max_timeout_seconds: int = 1800,
        local_mode: bool = False,
        custom_agents: list = None,
        server_id: str = None,
        agent_class: Type[AgentBase] = None,
        agent_args: tuple = (),
        agent_kwargs: dict = None,
    ) -> None:
        """Init a launcher of agent server.

        Args:
            host (`str`, defaults to `"localhost"`):
                Hostname of the agent server.
            port (`int`, defaults to `None`):
                Socket port of the agent server.
            max_pool_size (`int`, defaults to `8192`):
                Max number of task results that the server can accommodate.
            max_timeout_seconds (`int`, defaults to `1800`):
                Timeout for task results.
            local_mode (`bool`, defaults to `False`):
                Whether the started server only listens to local
                requests.
            custom_agents (`list`, defaults to `None`):
                A list of custom agent classes that are not in
                `agentscope.agents`.
            server_id (`str`, defaults to `None`):
                The id of the agent server. If not specified, a random id
                will be generated.
            agent_class (`Type[AgentBase]`, deprecated):
                The AgentBase subclass encapsulated by this wrapper.
            agent_args (`tuple`, deprecated): The args tuple used to
                initialize the agent_class.
            agent_kwargs (`dict`, deprecated): The args dict used to
                initialize the agent_class.
        """
        self.host = host
        self.port = check_port(port)
        self.max_pool_size = max_pool_size
        self.max_timeout_seconds = max_timeout_seconds
        self.local_mode = local_mode
        self.server = None
        self.stop_event = None
        self.parent_con = None
        self.custom_agents = custom_agents
        self.server_id = (
            self.generate_server_id() if server_id is None else server_id
        )
        if (
            agent_class is not None
            or len(agent_args) > 0
            or agent_kwargs is not None
        ):
            logger.warning(
                "`agent_class`, `agent_args` and `agent_kwargs` is deprecated"
                " in `AgentServerLauncher`",
            )

    def generate_server_id(self) -> str:
        """Generate server id"""
        return f"{self.host}:{self.port}-{_get_timestamp('%y%m%d-%H:%M:%S')}"

    def _launch_in_main(self) -> None:
        """Launch agent server in main-process"""
        logger.info(
            f"Launching agent server at [{self.host}:{self.port}]...",
        )
        asyncio.run(
            setup_agent_server_async(
                host=self.host,
                port=self.port,
                server_id=self.server_id,
                max_pool_size=self.max_pool_size,
                max_timeout_seconds=self.max_timeout_seconds,
                local_mode=self.local_mode,
                custom_agents=self.custom_agents,
            ),
        )

    def _launch_in_sub(self) -> None:
        """Launch an agent server in sub-process."""
        from agentscope._init import _INIT_SETTINGS

        self.stop_event = Event()
        self.parent_con, child_con = Pipe()
        start_event = Event()
        server_process = Process(
            target=setup_agent_server,
            kwargs={
                "host": self.host,
                "port": self.port,
                "server_id": self.server_id,
                "init_settings": _INIT_SETTINGS,
                "start_event": start_event,
                "stop_event": self.stop_event,
                "pipe": child_con,
                "max_pool_size": self.max_pool_size,
                "max_timeout_seconds": self.max_timeout_seconds,
                "local_mode": self.local_mode,
                "custom_agents": self.custom_agents,
            },
        )
        server_process.start()
        self.port = self.parent_con.recv()
        start_event.wait()
        self.server = server_process
        logger.info(
            f"Launch agent server at [{self.host}:{self.port}] success",
        )

    def launch(self, in_subprocess: bool = True) -> None:
        """launch an agent server.

        Args:
            in_subprocess (bool, optional): launch the server in subprocess.
                Defaults to True. For agents that need to obtain command line
                input, such as UserAgent, please set this value to False.
        """
        if in_subprocess:
            self._launch_in_sub()
        else:
            self._launch_in_main()

    def wait_until_terminate(self) -> None:
        """Wait for server process"""
        if self.server is not None:
            self.server.join()

    def shutdown(self) -> None:
        """Shutdown the agent server."""
        if self.server is not None:
            if self.stop_event is not None:
                self.stop_event.set()
                self.stop_event = None
            self.server.join()
            if self.server.is_alive():
                self.server.kill()
                logger.info(
                    f"Agent server at port [{self.port}] is killed.",
                )
            self.server = None


def as_server() -> None:
    """Launch an agent server with terminal command.

    Note:

        The arguments of `as_server` are listed as follows:

        * `--host`: the hostname of the server.
        * `--port`: the socket port of the server.
        * `--max_pool_size`: max number of task results that the server can
          accommodate.
        * `--max_timeout_seconds`: max timeout seconds of a task.
        * `--local_mode`: whether the started agent server only listens to
          local requests.

        In most cases, you only need to specify the `--host` and `--port`.

        .. code-block:: shell

            as_server --host localhost --port 12345

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="hostname of the server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12310,
        help="socket port of the server",
    )
    parser.add_argument(
        "--max_pool_size",
        type=int,
        default=8192,
        help="max number of task results that the server can accommodate",
    )
    parser.add_argument(
        "--max_timeout_seconds",
        type=int,
        default=1800,
        help="max timeout for task results",
    )
    parser.add_argument(
        "--local_mode",
        type=bool,
        default=False,
        help="whether the started agent server only listens to local requests",
    )
    args = parser.parse_args()
    launcher = AgentServerLauncher(
        host=args.host,
        port=args.port,
        max_pool_size=args.max_pool_size,
        max_timeout_seconds=args.max_timeout_seconds,
        local_mode=args.local_mode,
    )
    launcher.launch(in_subprocess=False)
    launcher.wait_until_terminate()
