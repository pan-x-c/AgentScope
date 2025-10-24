import asyncio
from typing import Dict
from pydantic import BaseModel, Field
from agentscope.model._trinity_model import LearnHybridChatModel, LearnTargetChatModel
from agentscope.model import DashScopeChatModel
from agentscope.agent import ReActAgent
from agentscope.tune.base_protocol import TaskContext, TrinityNativeLearnProtocol
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg

class DemoWorkflow__Math(TrinityNativeLearnProtocol):

    async def workflow_func(task_context: TaskContext) -> float:
        """Workflow function for ReAct agent training."""
        task = task_context.get_task()

        agent = ReActAgent(
            name="react_agent",
            sys_prompt="You are a helpful math problem solving agent.",
            model=task_context.get_chatmodel(
                DashScopeChatModel(model_name="qwen-max"),  # for debug
                "path-to-qwen2.5-14b-instruct"              # for training
            ),
            enable_meta_tool=True,
            formatter=OpenAIChatFormatter(),
        )

        class ResponseStructure(BaseModel):
            result: str = Field(description="Final answer to the math problem.")

        response = await agent.reply(
            msg=Msg("user", task["question"], role="user"),
            structured_model=ResponseStructure,
        )

        def calculate_reward(answer: str, truth: str) -> float:
            """Simple reward: 1.0 for exact match, else 0.0.

            This is a toy reward function; replace it with a more robust metric if needed.
            """
            return 1.0 if answer.strip() == truth.strip() else 0.0

        reward = calculate_reward(response.metadata["result"], task["answer"])
        task_context.set_workflow_result({"reward": reward})




# Choose a stage dev/train
__STAGE__ = 'debug-agentscope-workflow' # or 'train-agentscope-workflow'

# Develop mode, you can detach from trinity and implement your own TaskContext to simplify the debug process
if __STAGE__ == 'debug-agentscope-workflow':
    class DebugTaskContext(TaskContext):
        def get_task(self):
            return {
                "question": "What is 7 multiplied by 6?",
                "answer": "42"
            }

    async def debug():
        task_context = TaskContext()
        await DemoWorkflow__Math.workflow_func(task_context)
        reward = task_context.get_workflow_result()

    asyncio.run(debug())

# Train mode, attach to trinity, auto generate task context from trinity
elif __STAGE__ == 'train-agentscope-workflow':
    from agentscope.tune import learn
    DemoWorkflow__Math().learn()