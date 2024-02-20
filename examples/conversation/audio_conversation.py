# -*- coding: utf-8 -*-
"""A simple example for audio conversation between user and assistant agent."""
import agentscope
from agentscope.agents.audio_dialog_agent import AudioDialogAgent
from agentscope.agents.audio_user_agent import AudioUserAgent
from agentscope.pipelines.functional import sequentialpipeline

agentscope.init(
    model_configs=[
        {
            "model_type": "post_api_chat",
            "config_name": "gpt-3.5-turbo",
        },
    ],
)

# Init two agents
dialog_agent = AudioDialogAgent(
    name="Assistant",
    sys_prompt="你是一个人工智能助手。",
    model_config_name="gpt-3.5-turbo",  # replace by your model config name
)
user_agent = AudioUserAgent()

# start the conversation between user and assistant
x = None
while x is None or x.content != "退出。":
    x = sequentialpipeline([dialog_agent, user_agent], x)
