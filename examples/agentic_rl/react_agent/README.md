# Training Agent Workflow using RL with Trinity-RFT

> [Trinity-RFT](https://github.com/modelscope/Trinity-RFT) is a flexible, general-purpose framework for reinforcement fine-tuning (RFT) of large language models (LLMs). It provides high compatibility with agent frameworks and supports training agents without code modification.

AgentScope provides a `learn` interface to train agent workflows using reinforcement learning (RL). This interface leverages the functionality of Trinity-RFT, allowing users to train agents with minimal code changes.

---

## Overview

To train an agent using reinforcement learning, you need to define a workflow function with the following signature:

```python
def workflow_function(
    task: Dict,
    model: TrinityChatModel,
) -> float:
    """A workflow function that defines the agent's behavior and returns a reward signal.

    Args:
        task (Dict): The task to be performed by the agent workflow, typically containing input data and other relevant information.
        model (TrinityChatModel): The model used by the agent.

    Returns:
        float: The reward signal for the agent's performance on the task.
    """
```

Here, we use the `ReActAgent` to solve math problems as an example. Below is a simplified version of the training script. For the complete code, please refer to the [train_react.py](./train_react.py) file.


```python
from typing import Dict

from pydantic import BaseModel, Field

from agentscope.learn import learn, LearnConfig
from agentscope.model import TrinityChatModel
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg

class ResponseStructure(BaseModel):
    """Response structure for Math tasks.
    Note: This is a simplified response structure for demonstration purposes.
    """
    result: str = Field(
        description="Your final answer to the given math problem."
    )

def calculate_reward(answer: str, truth: str) -> float:
    """Calculate reward based on whether the prediction matches the answer.
    Note: This is a simplified reward function for demonstration purposes.
    """
    return 1.0 if answer.strip() == truth.strip() else 0.0

async def react_agent_workflow(task: Dict, model: TrinityChatModel) -> float:
    """The workflow function for the ReAct agent training.

    Args:
        task (Dict): The task to be performed by the agent workflow, here we suppose it contains 'question' and 'answer' keys.
        model (TrinityChatModel): The model used by the agent.
    """
    formatter = OpenAIChatFormatter()
    agent = ReActAgent(
        name="react_agent",
        sys_prompt=sys_prompt,
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
        workflow_func=react_agent_workflow,
        config=LearnConfig.load_config("/path/to/your/config.yaml")
    )
```

To convert an existing agent workflow into a trainable workflow, encapsulate your agent logic inside the workflow function and return a reward signal based on the agent's performance on the task.

---

## Preparation

1. **Hardware Requirements**
   - Ensure you have at least 2 NVIDIA GPUs with CUDA 12.4 or above installed.
   - This example uses 8 H20 GPUs for training, but you can adjust the configuration file (`gsm8k.yaml`) based on your hardware. For detailed configurations, please refer to the [Configuration Guide](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html).

2. **Install Trinity-RFT**
   - Follow the [Installation Guide](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_installation.html) to install the latest version of Trinity-RFT (>= 0.3.1).

3. **Download Required Resources**
   - Download the GSM8K dataset and Qwen/Qwen3-8B model checkpoints:

     ```bash
     huggingface-cli download openai/gsm8k --repo-type dataset
     huggingface-cli download Qwen/Qwen3-8B
     ```

---

## Run the Example

1. **Set up Ray Cluster**
   - Start the Ray cluster:

     ```bash
     ray start --head
     ```

2. **Run the Training Script**
   - Execute the training script:

     ```bash
     python train_react.py
     ```
