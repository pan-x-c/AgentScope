from typing import Dict

from pydantic import BaseModel, Field

from agentscope.tune import tune, TuneConfig
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
    This structure makes the agent's output easy to parse,
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
    tune(
        workflow_func=react_workflow_function,
        config=TuneConfig.load_config("/path/to/your/config.yaml"),
    )