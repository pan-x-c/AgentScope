# -*- coding: utf-8 -*-
"""A general dialog agent."""
from typing import Optional, Union
import os

import dashscope
from uuid import uuid4
from dashscope.audio.tts import SpeechSynthesizer
from loguru import logger
from playsound import playsound

from ..message import Msg
from .agent import AgentBase
from ..file_manager import file_manager
from ..prompt import PromptEngine
from ..prompt import PromptType


class AudioDialogAgent(AgentBase):
    """A simple agent used to perform a dialogue. Your can set its role by
    `sys_prompt`."""

    def __init__(
        self,
        name: str,
        sys_prompt: Optional[str] = None,
        model_config_name: str = None,
        use_memory: bool = True,
        memory_config: Optional[dict] = None,
        prompt_type: Optional[PromptType] = PromptType.LIST,
    ) -> None:
        """Initialize the dialog agent.

        Arguments:
            name (`str`):
                The name of the agent.
            sys_prompt (`Optional[str]`):
                The system prompt of the agent, which can be passed by args
                or hard-coded in the agent.
            model_config_name (`str`, defaults to None):
                The name of the model config, which is used to load model from
                configuration.
            use_memory (`bool`, defaults to `True`):
                Whether the agent has memory.
            memory_config (`Optional[dict]`):
                The config of memory.
            prompt_type (`Optional[PromptType]`, defaults to
            `PromptType.LIST`):
                The type of the prompt organization, chosen from
                `PromptType.LIST` or `PromptType.STRING`.
        """
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            use_memory=use_memory,
            memory_config=memory_config,
        )

        # init prompt engine
        self.engine = PromptEngine(self.model, prompt_type=prompt_type)

    def reply(self, x: dict = None) -> dict:
        """Reply function of the agent. Processes the input data,
        generates a prompt using the current dialogue memory and system
        prompt, and invokes the language model to produce a response. The
        response is then formatted and added to the dialogue memory.

        Args:
            x (`dict`, defaults to `None`):
                A dictionary representing the user's input to the agent. This
                input is added to the dialogue memory if provided. Defaults to
                None.
        Returns:
            A dictionary representing the message generated by the agent in
            response to the user's input.
        """
        # record the input if needed
        if x is not None:
            self.memory.add(x)

        # prepare prompt
        prompt = self.engine.join(
            self.sys_prompt,
            self.memory.get_memory(),
        )

        # call llm and generate response
        response = self.model(prompt).text
        msg = Msg(self.name, response)

        # Print/speak the message in this agent's voice
        self.speak(msg)

        # Record the message in memory
        self.memory.add(msg)

        return msg

    def text2audio(self, text: Union[str, dict]) -> None:
        """convert text to audio"""
        # dashscope.api_key = ""
        result = SpeechSynthesizer.call(
            model="sambert-zhichu-v1",
            text=text,
            sample_rate=22050,
            format="wav",
        )
        logger.info(f"convert [{text}] to audio")
        audio_path = os.path.join(file_manager.dir_file, f"{uuid4().hex}.wav")
        if result.get_audio_data() is not None:
            with open(audio_path, "wb") as f:
                f.write(result.get_audio_data())
        playsound(audio_path)

    def speak(self, content: Union[str, dict]) -> None:
        self.text2audio(content["content"])
