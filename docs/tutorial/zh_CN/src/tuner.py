# -*- coding: utf-8 -*-
"""
.. _tuner:

Tuner
=================

AgentScope 提供了 ``tuner`` 模块用于通过强化学习 (RL) 训练智能体应用。
本教程将指导你如何使用 ``tuner`` 模块提升智能体在特定任务上的表现，包括：

- 介绍 ``tuner`` 模块的主要组件。
- 演示如何实现调优流程所需的必要代码组件。
- 展示如何配置和运行调优流程。

主要组件
~~~~~~~~~~~~~~~~~~~
``tuner`` 模块引入了训练智能体工作流所需的三个主要组件：

- **任务数据集**：用于训练和评估智能体应用的任务集合。
- **工作流函数**：内部包含了被调优的智能体应用。
- **评判函数**：用于评估智能体在特定任务上的表现，并为调优过程提供奖励信号的函数。

除了这些组件，``tuner`` 模块还提供了一些用于自定义调优过程的配置类，包括：

- **TunerChatModel**：仅用于调优的可配置对话模型，完全兼容 AgentScope 的 ``OpenAIChatModel``。
- **Algorithm**：用于调优的强化学习算法，例如 GRPO、PPO 等。

如何实现
~~~~~~~~~~~~~~~~~~~
下面我们将实现一个可以通过 ``tuner`` 模块训练的简单数学智能体应用。

任务数据集
--------------------
任务数据集包含用于训练和评估智能体应用的任务集合。

数据集应采用 huggingface `datasets <https://huggingface.co/docs/datasets/quickstart>`_ 格式，并可通过 ``datasets.load_dataset`` 函数加载。例如：

.. code-block:: text

    my_dataset/
        ├── train.jsonl  # 训练样本
        └── test.jsonl   # 测试样本

假设你的 `train.jsonl` 包含如下样本：

.. code-block:: json

    {"question": "2 + 2 等于多少？", "answer": "4"}
    {"question": "4 + 4 等于多少？", "answer": "8"}

工作流函数
--------------------
工作流函数定义了智能体如何与环境交互并做出决策。所有工作流函数应遵循 ``agentscope.tuner.WorkflowType`` 定义的输入/输出签名。

下面是一个使用 ReAct 智能体回答数学问题的简单工作流函数示例。
"""

from typing import Dict, Optional
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import ChatModelBase
from agentscope.tuner import WorkflowOutput


async def example_workflow_function(
    task: Dict,
    model: ChatModelBase,
    auxiliary_models: Optional[Dict[str, ChatModelBase]] = None,
) -> WorkflowOutput:
    """一个用于调优的工作流函数示例。

    Args:
        task (dict): 任务信息。
        model (ChatModelBase): 智能体使用的对话模型。
        auxiliary_models (Optional[Dict[str, ChatModelBase]]):
            用于辅助的额外对话模型，一般用于多智能体场景下模拟其他非训练智能体的行为。

    Returns:
        WorkflowOutput: 工作流生成的输出。
    """
    agent = ReActAgent(
        name="react_agent",
        sys_prompt="你是一个善于解决数学问题的智能体。",
        model=model,
        formatter=OpenAIChatFormatter(),
    )

    response = await agent.reply(
        msg=Msg(
            "user",
            task["question"],
            role="user",
        ),  # 从任务中提取问题
    )

    return WorkflowOutput(  # 将响应放入 WorkflowOutput
        response=response,
    )


# %%
# 你可以直接用任务字典和对话模型运行此工作流函数。
# 例如：

import asyncio
import os
from agentscope.model import DashScopeChatModel

task = {"question": "123 加 456 等于多少？", "answer": "579"}
model = DashScopeChatModel(
    model_name="qwen-max",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)
workflow_output = asyncio.run(example_workflow_function(task, model))
assert isinstance(
    workflow_output.response,
    Msg,
), "在此示例中，响应应为 Msg 实例。"
print("\n工作流响应:", workflow_output.response.get_text_content())

# %%
#
# 评判函数
# --------------------
# 评判函数用于评估智能体在特定任务上的表现，并为调优过程提供奖励信号。
# 所有评判函数应遵循 ``agentscope.tuner.JudgeType`` 定义的输入/输出签名。
# 下面是一个简单的评判函数示例，通过比较智能体响应与标准答案。

from typing import Any
from agentscope.tuner import JudgeOutput


async def example_judge_function(
    task: Dict,
    response: Any,
    auxiliary_models: Optional[Dict[str, ChatModelBase]] = None,
) -> JudgeOutput:
    """仅用于演示的简单评判函数。

    Args:
        task (Dict): 任务信息。
        response (Any): WorkflowOutput 的响应字段。
        auxiliary_models (Optional[Dict[str, ChatModelBase]]):
            用于 LLM-as-a-Judge 的辅助模型。
    """
    ground_truth = task["answer"]
    reward = 1.0 if ground_truth in response.get_text_content() else 0.0
    return JudgeOutput(reward=reward)


judge_output = asyncio.run(
    example_judge_function(
        task,
        workflow_output.response,
    ),
)
print(f"评判奖励: {judge_output.reward}")

# %%
#
# .. tip:: 你可以在评判函数中利用已有的 `MetricBase <https://github.com/agentscope-ai/agentscope/blob/main/src/agentscope/evaluate/_metric_base.py>`_ 实现来计算更复杂的指标，并将它们组合成复合奖励。
#
# 如何运行
# ~~~~~~~~~~~~~~~
# 最后，你可以使用 ``tuner`` 模块配置并运行调优过程。
# 在开始调优前，请确保你的环境已安装 `Trinity-RFT <https://github.com/modelscope/Trinity-RFT>`_，这是 ``tuner`` 模块的依赖。
#
# 下面是配置和启动调优过程的示例。
#
# .. note:: 此示例仅供演示。完整可运行示例请参考 `Tune ReActAgent <https://github.com/agentscope-ai/agentscope/tree/main/examples/tuner/react_agent>`_
#
# .. code-block:: python
#
#        from agentscope.tuner import tune, Algorithm, Dataset, TunerChatModel
#        # 你的工作流 / 评判函数 ...
#
#        if __name__ == "__main__":
#            dataset = Dataset(path="my_dataset", split="train")
#            model = TunerChatModel(model_path="Qwen/Qwen3-0.6B", max_model_len=16384)
#            algorithm = Algorithm(
#                algorithm_type="multi_step_grpo",
#                group_size=8,
#                batch_size=32,
#                learning_rate=1e-6,
#            )
#            tune(
#                workflow_func=example_workflow_function,
#                judge_func=example_judge_function,
#                model=model,
#                train_dataset=dataset,
#                algorithm=algorithm,
#            )
#
# 这里我们用 ``Dataset`` 配置训练数据集，用 ``TunerChatModel`` 初始化可训练模型，用 ``Algorithm`` 指定强化学习算法及其超参数。
#
# .. tip::
#   ``tune`` 函数基于 `Trinity-RFT <https://github.com/modelscope/Trinity-RFT>`_ 实现，并在内部将输入参数转换为 YAML 配置。
#   高级用户可以忽略 ``model``、``train_dataset``、``algorithm`` 参数，改为通过 ``config_path`` 参数提供指向 YAML 文件的配置路径。
#   我们推荐使用配置文件方式以便对训练过程进行更细粒度的控制，从而利用 Trinity-RFT 提供的高级功能。
#   你可以参考 Trinity-RFT 的 `配置指南 <https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html>`_ 了解更多配置选项。
#
# 你可以将上述代码保存为 ``main.py``，并用如下命令运行：
#
# .. code-block:: bash
#
#        ray start --head
#        python main.py
#
# 检查点和日志会自动保存到当前工作目录下的 ``checkpoints/AgentScope`` 目录，每次运行会以时间戳为后缀保存到子目录。
# tensorboard 日志可在检查点目录下的 ``monitor/tensorboard`` 中找到。
#
# .. code-block:: text
#
#        your_workspace/
#            └── checkpoints/
#                └──AgentScope/
#                    └── Experiment-20260104185355/  # 每次运行以时间戳保存
#                        ├── monitor/
#                        │   └── tensorboard/  # tensorboard 日志
#                        └── global_step_x/    # 第 x 步保存的模型检查点
#
