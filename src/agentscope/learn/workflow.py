# -*- coding: utf-8 -*-
"""Workflow for agent learning."""

from typing import (
    Dict,
    Callable,
    Awaitable,
    Optional,
    List,
    TYPE_CHECKING,
)

from ..model import TrinityChatModel

if TYPE_CHECKING:
    from openai import OpenAI
    from trinity.common.workflow import Workflow as TrinityWorkflow
    from trinity.common.workflow import Task, WORKFLOWS
    from trinity.common.model import ModelWrapper
    from trinity.common.experience import Experience
else:
    Task = "trinity.common.workflows.Task"
    TrinityWorkflow = "trinity.common.workflows.Workflow"
    ModelWrapper = "trinity.common.model.ModelWrapper"
    Experience = "trinity.common.experience.Experience"
    OpenAI = "openai.OpenAI"


TRAINABLE_WORKFLOW_NAME = "agentscope_trainable_workflow"


WorkflowType = Callable[[Dict, TrinityChatModel], Awaitable[float]]


@WORKFLOWS.register_module(TRAINABLE_WORKFLOW_NAME)
class TrinityWorkflowAdapter(TrinityWorkflow):
    """Adapter to wrap a Workflow instance for Trinity compatibility."""

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[OpenAI]],
    ):
        """Initialize the adapter with the task and model."""
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.workflow_func = task.workflow_args.get("workflow_func")
        self.model: TrinityChatModel = TrinityChatModel(
            model.get_openai_async_client(),
        )

    @property
    def asynchronous(self) -> bool:
        """This workflow runs asynchronously."""
        return True

    @property
    def repeatable(self) -> bool:
        """This workflow is not repeatable."""
        return False

    @property
    def resetable(self) -> bool:
        """This workflow cannot be reset."""
        return False

    def construct_experiences(
        self,
        reward: float,
    ) -> List[Experience]:
        """Construct experiences from the agent's interaction history.

        Args:
            reward (float): The reward value to assign to each experience.

        Returns:
            List: A list of Experience objects.
        """
        exps = self.model.extract_experience_from_history()
        for exp in exps:
            exp.reward = reward
        return exps

    async def run_async(self) -> List[Experience]:
        """Run the workflow asynchronously and return experiences."""
        reward = await self.workflow_func(self.task.raw_task, self.model)
        return self.construct_experiences(reward)
