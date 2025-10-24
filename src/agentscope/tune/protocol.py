# -*- coding: utf-8 -*-
"""The configurations for agent learning."""

from dataclasses import dataclass
from pydantic import BaseModel, Field
from agentscope.model import DashScopeChatModel
from typing import List
from agentscope.tune.workflow import validate_function_signature


class TaskContext:
    # define task context during ** workflow development **, no training consideration is needed here.
    def __init__(self, mode='debug'):
        self.workflow_result = None
        self.mode = mode  # 'debug' or 'train'

    def get_task(self):
        raise NotImplementedError

    def get_chatmodel(self, debug_model, train_model):
        if self.mode=='debug':
            return DashScopeChatModel(model_name=debug_model)
        else:
            return train_model

    def get_workflow_result(self):
        return self.workflow_result

    def set_workflow_result(self, result):
        self.workflow_result = result



class LearnProtocol(BaseModel):  # type: ignore [no-redef]
    # trainer selection
    trainer: str = Field(default="trinity-native")
    # agentflow name in trainer
    agentflow_name: str = Field(default="agentscope-evolver")
    # trainer's config path
    trainer_config_path: str = Field(default="")
    # [reserved] trainable targets defined in agentflow
    trainable_targets: List[str] = Field(default=[])
    # [reserved] Use dataset provided by the trainer
    # (True: read each query from workflow input; False: AgentScope handles each query)
    external_dataset: bool = Field(default=True)
    # [reserved] Use external environment provided by the trainer
    # (True: read environment handle from input; False: AgentScope runs environment and tools)
    external_environment: bool = Field(default=False)
    # [reserved] Use external reward provided by the trainer
    # (True: compute reward outside AgentScope after workflow; False: AgentScope computes reward)
    external_reward: bool = Field(default=False)

    async def workflow_func(self, task_context: TaskContext) -> None:
        """define workflow here"""
        raise NotImplementedError

    def learn(self):
        """implement learning launcher here"""
        raise NotImplementedError



class TrinityNativeLearnProtocol(LearnProtocol):

    def learn(self):
        """train adapter for trinity-native trainer"""
        from agentscope.tune import tune, TuneConfig
        from typing import Dict
        from agentscope.model import TrinityChatModel
        def react_workflow_function(task: Dict, trinity_models: TrinityChatModel):
            task_context = TaskContext(mode='train')
            task_context.get_task = lambda: task
            task_context.get_chatmodel = lambda debug_model, train_model: trinity_models.select(train_model)
            self.workflow_func(task_context)
            return task_context.get_workflow_result()['reward']

        if not validate_function_signature(react_workflow_function):
            raise ValueError("The workflow function signature is invalid.")

        tune(
            workflow_func=react_workflow_function,
            config=TuneConfig.load_config(self.trainer_config_path),
        )


class AgentEvoTrinityLearnProtocol(LearnProtocol):
    ...