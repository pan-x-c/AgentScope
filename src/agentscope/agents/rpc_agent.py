# -*- coding: utf-8 -*-
""" Base class for Rpc Agent """
from typing import Type, Optional, Union, Sequence

from agentscope.agents.agent import AgentBase
from agentscope.message import (
    PlaceholderMessage,
    serialize,
)
from agentscope.rpc import RpcAgentClient
from agentscope.server.launcher import RpcAgentServerLauncher
from agentscope.studio._client import _studio_client


class RpcAgent(AgentBase):
    """A wrapper to extend an AgentBase into a gRPC Client."""

    def __init__(
        self,
        name: str,
        host: str = "localhost",
        port: int = None,
        agent_class: Type[AgentBase] = None,
        agent_configs: Optional[dict] = None,
        max_pool_size: int = 8192,
        max_timeout_seconds: int = 1800,
        local_mode: bool = True,
        lazy_launch: bool = True,
        agent_id: str = None,
        connect_existing: bool = False,
        upload_source_code: bool = False,
    ) -> None:
        """Initialize a RpcAgent instance.

        Args:
            name (`str`): the name of the agent.
            host (`str`, defaults to `localhost`):
                Hostname of the rpc agent server.
            port (`int`, defaults to `None`):
                Port of the rpc agent server.
            agent_class (`Type[AgentBase]`):
                the AgentBase subclass of the source agent.
            agent_configs (`dict`): The args used to
                init configs of the agent, generated by `_AgentMeta`.
            max_pool_size (`int`, defaults to `8192`):
                Max number of task results that the server can accommodate.
            max_timeout_seconds (`int`, defaults to `1800`):
                Timeout for task results.
            local_mode (`bool`, defaults to `True`):
                Whether the started gRPC server only listens to local
                requests.
            lazy_launch (`bool`, defaults to `True`):
                Only launch the server when the agent is called.
            agent_id (`str`, defaults to `None`):
                The agent id of this instance. If `None`, it will
                be generated randomly.
            connect_existing (`bool`, defaults to `False`):
                Set to `True`, if the agent is already running on the agent
                server.
            upload_source_code (`bool`, default to `False`):
                Upload the source code of the agent to the agent server.
                Only takes effect when connecting to an existing server.
                When you are using an agent that doens't exist on the server
                (such as your customized agent that is not officially provided
                by AgentScope), please set this value to `True`.
        """
        super().__init__(name=name)
        self.agent_class = agent_class
        self.agent_configs = agent_configs
        self.host = host
        self.port = port
        self.server_launcher = None
        self.client = None
        self.connect_existing = connect_existing
        if agent_id is not None:
            self._agent_id = agent_id
        # if host and port are not provided, launch server locally
        launch_server = port is None
        if launch_server:
            self.host = "localhost"
            studio_url = None
            if _studio_client.active:
                studio_url = _studio_client.studio_url
            self.server_launcher = RpcAgentServerLauncher(
                host=self.host,
                port=port,
                max_pool_size=max_pool_size,
                max_timeout_seconds=max_timeout_seconds,
                local_mode=local_mode,
                custom_agents=[agent_class],
                studio_url=studio_url,
            )
            if not lazy_launch:
                self._launch_server()
        else:
            self.client = RpcAgentClient(
                host=self.host,
                port=self.port,
                agent_id=self.agent_id,
            )
            if not self.connect_existing:
                self.client.create_agent(
                    agent_configs,
                    upload_source_code=upload_source_code,
                )

    def _launch_server(self) -> None:
        """Launch a rpc server and update the port and the client"""
        self.server_launcher.launch()
        self.port = self.server_launcher.port
        self.client = RpcAgentClient(
            host=self.host,
            port=self.port,
            agent_id=self.agent_id,
        )
        self.client.create_agent(self.agent_configs)

    def reply(self, x: dict = None) -> dict:
        if self.client is None:
            self._launch_server()
        return PlaceholderMessage(
            name=self.name,
            content=None,
            client=self.client,
            x=x,
        )

    def observe(self, x: Union[dict, Sequence[dict]]) -> None:
        if self.client is None:
            self._launch_server()
        self.client.call_agent_func(
            func_name="_observe",
            value=serialize(x),  # type: ignore[arg-type]
        )

    def clone_instances(
        self,
        num_instances: int,
        including_self: bool = True,
    ) -> Sequence[AgentBase]:
        """
        Clone a series of this instance with different agent_id and
        return them as a list.

        Args:
            num_instances (`int`): The number of instances in the returned
            list.
            including_self (`bool`): Whether to include the instance calling
            this method in the returned list.

        Returns:
            `Sequence[AgentBase]`: A list of agent instances.
        """
        generated_instance_number = (
            num_instances - 1 if including_self else num_instances
        )
        generated_instances = []

        # launch the server before clone instances
        if self.client is None:
            self._launch_server()

        # put itself as the first element of the returned list
        if including_self:
            generated_instances.append(self)

        # clone instances without agent server
        for _ in range(generated_instance_number):
            new_agent_id = self.client.clone_agent(self.agent_id)
            generated_instances.append(
                RpcAgent(
                    name=self.name,
                    host=self.host,
                    port=self.port,
                    agent_id=new_agent_id,
                    connect_existing=True,
                ),
            )
        return generated_instances

    def stop(self) -> None:
        """Stop the RpcAgent and the rpc server."""
        if self.server_launcher is not None:
            self.server_launcher.shutdown()

    def __del__(self) -> None:
        self.stop()
