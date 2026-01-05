# -*- coding: utf-8 -*-
# pylint: disable=too-many-statements
"""Unit tests for Trinity-RFT model class."""
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import Mock, AsyncMock
from typing import Dict

from agentscope.model import ChatResponse
from agentscope.message import TextBlock
from agentscope.tuner import TunerChatModel, WorkflowOutput, JudgeOutput
from agentscope.tuner._config import check_judge_function, check_workflow_function


class TestTunerChatModel(IsolatedAsyncioTestCase):
    """Test cases for TunerChatModel."""

    async def test_init_with_trinity_client(self) -> None:
        """Test initialization with a valid OpenAI async client."""
        MODEL_NAME = "Qwen/Qwen3-1.7B"
        mock_client = Mock()
        mock_client.model_path = MODEL_NAME

        # test init
        model_1 = TunerChatModel(
            model_path=MODEL_NAME,
            max_model_len=16384,
            enable_thinking=False,
            temperature=0.8,
            tensor_parallel_size=2,
            inference_engine_num=2,
        )
        model_2 = TunerChatModel(
            model_path=MODEL_NAME,
            max_model_len=16384,
            enable_thinking=True,
            max_tokens=500,
            top_p=0.9,
        )
        model_1.set_openai_client(mock_client)
        model_2.set_openai_client(mock_client)
        self.assertEqual(model_1.model_name, MODEL_NAME)
        self.assertFalse(model_1.stream)
        self.assertIs(model_1.client, mock_client)
        self.assertEqual(model_2.model_name, MODEL_NAME)
        self.assertFalse(model_2.stream)
        self.assertIs(model_2.client, mock_client)

        # create mock response
        messages = [{"role": "user", "content": "Hello"}]
        mock_message = Mock()
        mock_message.content = "Hi there!"
        mock_message.reasoning_content = None
        mock_message.tool_calls = []
        mock_message.audio = None
        mock_message.parsed = None
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_response.usage = mock_usage

        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_response,
        )

        result = await model_1(messages)
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], MODEL_NAME)
        self.assertEqual(call_args["messages"], messages)
        self.assertFalse(call_args["stream"])
        self.assertFalse(call_args["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(call_args["temperature"], 0.8)
        self.assertIsInstance(result, ChatResponse)
        expected_content = [
            TextBlock(type="text", text="Hi there!"),
        ]
        self.assertEqual(result.content, expected_content)

        result = await model_2(messages)
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], MODEL_NAME)
        self.assertEqual(call_args["messages"], messages)
        self.assertFalse(call_args["stream"])
        self.assertTrue(call_args["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(call_args["max_tokens"], 500)
        self.assertEqual(call_args["top_p"], 0.9)
        self.assertIsInstance(result, ChatResponse)
        expected_content = [
            TextBlock(type="text", text="Hi there!"),
        ]
        self.assertEqual(result.content, expected_content)


async def correct_workflow_func(
    task: Dict, model: TunerChatModel, auxiliary_models: Dict[str, TunerChatModel],
) -> WorkflowOutput:
    """Correct interface matching the workflow type."""
    return WorkflowOutput(
        response="Test response",
    )

async def correct_workflow_func_no_aux(
    task: Dict, model: TunerChatModel,
) -> WorkflowOutput:
    """Correct interface matching the workflow type without auxiliary models."""
    return WorkflowOutput(
        response="Test response",
    )

async def incorrect_workflow_func_1(task: Dict) -> WorkflowOutput:
    """Incorrect interface not matching the workflow type."""
    return WorkflowOutput(
        response="Test response",
    )


async def incorrect_workflow_func_2(
    task: Dict, model: TunerChatModel, aux_model: int
) -> WorkflowOutput:
    """Incorrect interface not matching the workflow type."""
    return WorkflowOutput(
        response="Test response",
    )


async def correct_judge_func(
    task: Dict,
    workflow_output: WorkflowOutput,
    auxiliary_models: Dict[str, TunerChatModel],
) -> JudgeOutput:
    """Correct interface matching the judge type."""
    return JudgeOutput(
        reward=1.0,
    )

async def incorrect_judge_func_1(
    wrong_name: Dict,
    workflow_output: WorkflowOutput,
) -> JudgeOutput:
    """Incorrect interface not matching the judge type."""
    return JudgeOutput(
        reward=1.0,
    )

async def incorrect_judge_func_2(
    workflow_output: WorkflowOutput,
) -> JudgeOutput:
    """Incorrect interface not matching the judge type."""
    return JudgeOutput(
        reward=1.0,
    )


class TestTunerFunctionType(IsolatedAsyncioTestCase):
    """Test cases for tuner function type validation."""

    def test_validate_workflow_type(self) -> None:
        """Test workflow type validation."""
        # Correct cases
        check_workflow_function(correct_workflow_func)
        check_workflow_function(correct_workflow_func_no_aux)

        # Incorrect cases
        with self.assertRaises(ValueError):
            check_workflow_function(incorrect_workflow_func_1)
        with self.assertRaises(ValueError):
            check_workflow_function(incorrect_workflow_func_2)

        # Correct cases
        check_judge_function(correct_judge_func)

        # Incorrect cases
        with self.assertRaises(ValueError):
            check_judge_function(incorrect_judge_func_1)
        with self.assertRaises(ValueError):
            check_judge_function(incorrect_judge_func_2)
