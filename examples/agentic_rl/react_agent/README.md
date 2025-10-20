# Training agent workflows with RL using Trinity-RFT

AgentScope exposes a `learn` interface to train agent workflows using reinforcement learning (RL).
The `learn` interface leverages [Trinity-RFT](https://github.com/modelscope/Trinity-RFT) which supports training agents with minimal code changes.

---

## How to implement

To train an agent workflow using RL, implement a workflow function with the
following signature:

```python
def workflow_function(
    task: Dict,
    model: TrinityChatModel,
) -> float:
    """Run the agent workflow on a single task and return a scalar reward.

    Args:
        task (Dict): Input data for the workflow (for example, contains 'question' and
            'answer' keys for a math problem).
        model (TrinityChatModel): The model instance used by the agent.

    Returns:
        float: A reward signal measuring the agent's performance on the task.
    """
```

Here we use a math problem solving scenario as an example to illustrate how to convert an existing agent workflow into a trainable workflow function.

Suppose you have an agent workflow that solves math problems using the `ReActAgent`

```python
from agentscope.agent import ReActAgent

# model = ...  # Initialize your ChatModel here

query = "What is the sum of the first 10 prime numbers?"
agent = ReActAgent(
    name="react_agent",
    sys_prompt="You are a helpful math problem solving agent.",
    model=model,
    enable_meta_tool=True,
    formatter=OpenAIChatFormatter(),
)

response = await agent.reply(
    msg=Msg("user", query, role="user"),
)

print(response)
```

To convert the above agent workflow into a trainable workflow function, there are 4 main steps:

1. Define the workflow function with the required signature (`react_workflow_function`).
2. Initialize the agent and run it with given `task` and `model`.
3. Implement a reward calculation mechanism based on the agent's response (`calculate_reward` and `ResponseStructure`).
4. Use the `learn` interface to train the workflow function.


```python
from typing import Dict

from pydantic import BaseModel, Field

from agentscope.learn import learn, LearnConfig
from agentscope.model import TrinityChatModel
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg


def calculate_reward(answer: str, truth: str) -> float:
    """Simple reward: 1.0 for exact match, else 0.0.

    This is a toy reward function; replace it with a more robust metric if needed.
    """

    return 1.0 if answer.strip() == truth.strip() else 0.0


class ResponseStructure(BaseModel):
    """Response structure for math tasks (simplified).
    This structure let the agent output be easily parsed,
    allowing for easy reward calculation.
    """

    result: str = Field(description="Final answer to the math problem.")


async def react_workflow_function(task: Dict, model: TrinityChatModel) -> float:
    """Workflow function for ReAct agent training."""

    agent = ReActAgent(
        name="react_agent",
        sys_prompt="You are a helpful math problem solving agent.",
        model=model,
        enable_meta_tool=True,
        formatter=OpenAIChatFormatter(),
    )

    response = await agent.reply(
        msg=Msg("user", task["question"], role="user"),
        structured_model=ResponseStructure,
    )

    reward = calculate_reward(response.metadata["result"], task["answer"])
    return reward


if __name__ == "__main__":
    learn(
        workflow_func=react_workflow_function,
        config=LearnConfig.load_config("/path/to/your/config.yaml"),
    )
```

> Above code is a simplified example.
> For a complete implementation, see [train_react.py](./train_react.py).
> For configuration details, see [Trinity-RFT Configuration Guide](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html).

---

## How to run

After implementing the workflow function, follow these steps to run the training:

1. Prerequisites

    - At least 2 NVIDIA GPUs with CUDA 12.4 or newer.
    - Adjust the configuration file ([gsm8k.yaml](./gsm8k.yaml)) based on your hardware.
    - Follow the Trinity-RFT [installation guide](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_installation.html) to install a compatible version (Trinity-RFT >= 0.3.1).
    - Download the GSM8K dataset and Qwen/Qwen3-8B model checkpoints (example):

      ```bash
      huggingface-cli download openai/gsm8k --repo-type dataset
      huggingface-cli download Qwen/Qwen3-8B
      ```

2. Set up a Ray cluster

    ```bash
    ray start --head
    # for multi-node setup, run the following command on worker nodes
    # ray start --address=<master_address>
    ```

3. Run the training script

    ```bash
    python train_react.py
    ```
