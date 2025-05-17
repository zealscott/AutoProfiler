# -*- coding: utf-8 -*-
from typing import Any

from loguru import logger
import json

from agentscope.exception import ResponseParsingError
from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.parsers import MarkdownJsonDictParser

from util.prompt_loader import *
from util.data_loader import SafeDict
from util.parsing import parsing_function_response

from agentscope.utils.token_utils import count_openai_token


class Profiler(AgentBase):
    """An agent class that used to plan, analysis, reason for profiling attack.

    Note that the `Profiler` agent is only used for reasoning. It do not store any data and do not have any tools.
    """

    def __init__(
        self,
        name: str,
        model_config_name: str,
        target_attributes: list,
        sys_prompt: str = "You're a helpful analyst. Your name is {name}.",
        count_token: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the ReAct agent with the given name, model config name
        and tools.

        Args:
            name (`str`):
                The name of the agent.
            sys_prompt (`str`):
                The system prompt of the agent.
            model_config_name (`str`):
                The name of the model config, which is used to load model from
                configuration.
        """
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )

        self.count_token = count_token
        self.model.max_retries = 20

        if not sys_prompt.endswith("\n"):
            sys_prompt = sys_prompt + "\n"

        self.sys_prompt = "\n".join(
            [
                sys_prompt.format(name=self.name),
                PROFILER_SYS_PROMPT.format_map(
                    SafeDict(
                        attributes=", ".join(target_attributes),
                        attributes_info=attr_converter(target_attributes, "list"),
                    )
                ),
            ],
        )

        map_attr = SafeDict(
            attributes=", ".join(target_attributes),
            attributes_info=attr_converter(target_attributes, "string"),
        )
        self.think_prompt = PROFILER_THINK_PROMPT.format_map(map_attr)
        self.reason_prompt = PROFILER_REASON_PROMPT.format_map(map_attr)

        # Put sys prompt into memory
        self.memory.add(Msg("system", self.sys_prompt, role="system"))

        self.think_parser = MarkdownJsonDictParser(
            content_hint={
                "think": "what do you think about the situation",
                "action": "what to do next (reason|retrieval|search|finish)",
                "instruction": "the command for next action",
            },
            required_keys=["think", "action", "instruction"],
        )

        self.reason_parser = MarkdownJsonDictParser(
            content_hint={
                "think": "what do you think about the situation",
                "results": str(
                    [
                        {
                            "type": "{attribute name}",
                            "confidence": "{confidence score}",
                            "evidence": "{clue for guessing}",
                            "guess": "{inferred information}",
                        }
                    ]
                ),
            },
            required_keys=["results"],
        )

    def reply(self, x: dict = None) -> dict:
        raise NotImplementedError

    def think(self, x: dict = None, reset=False) -> dict:
        if reset:
            # clear all previous memory
            self.memory.clear()
            # Put sys prompt into memory
            self.memory.add(Msg("system", self.sys_prompt, role="system"))

        # store the inferred PIIs
        self.memory.add(x)
        # we only store the last ouput of this iteration reasoning
        start_mem_idx = self.memory.size()
        # The agent will keep parsing the response until it is parsed successfully
        while True:
            self.speak(f" [PROFILER] THINK ".center(70, "#"))

            # Prepare hint to remind model what the response format is
            # Won't be recorded in memory to save tokens
            hint_msg = Msg(
                "system",
                self.think_parser.format_instruction,
                role="system",
            )

            # Prepare prompt for the model
            prompt = self.model.format(
                self.memory.get_memory(), Msg("system", self.think_prompt, role="system"), hint_msg
            )

            # Generate and parse the response
            try:
                if self.count_token:
                    n_tokens = count_openai_token(prompt, model="gpt-4-0613")
                    self.speak(f" Count input token {n_tokens} ".center(70, "#"))
                res = self.model(
                    prompt,
                    parse_func=self.think_parser.parse,
                    max_retries=1,
                )

                end_mem_idx = self.memory.size()
                if end_mem_idx != start_mem_idx:
                    # delete the memory from the last iteration (wrong parsing)
                    delete_idx = list(range(start_mem_idx, end_mem_idx))
                    self.memory.delete(delete_idx)

                # Record the response in memory
                msg_response = Msg(self.name, res.text, "assistant")
                self.memory.add(msg_response)

                # Print out the response
                self.speak(msg_response)
                # Break the loop if the response is parsed successfully
                return res
            except ResponseParsingError as e:
                # Print out raw response from models for developers to debug
                response_msg = Msg(self.name, e.raw_response, "assistant")
                self.speak(response_msg)

                if "context_length_exceeded" in e.raw_response:
                    print(f"tackling context_length_exceeded error: {e.raw_response}")
                    error_msg = Msg(
                        "system",
                        "The response is too long. Only provide necessary information.",
                        "system",
                    )
                    # clear the memory
                    delete_idx = list(range(2, self.memory.size()))
                    self.memory.delete(delete_idx)
                    self.memory.add([error_msg])

                # Re-correct by model itself
                error_msg = Msg("system", str(e), "system")
                self.speak(error_msg)

                self.memory.add([response_msg, error_msg])

    def reason(self, x: dict = None) -> dict:
        # change role to reason
        self.memory.add(x)
        # we only store the last ouput of this iteration reasoning
        start_mem_idx = self.memory.size()
        # The agent will keep parsing the response until it is parsed successfully
        while True:
            self.speak(f" [PROFILER] REASON ".center(70, "#"))

            # Prepare hint to remind model what the response format is
            # Won't be recorded in memory to save tokens
            hint_msg = Msg(
                "system",
                self.reason_parser.format_instruction,
                role="system",
            )

            # Prepare prompt for the model
            prompt = self.model.format(
                self.memory.get_memory(), Msg("system", self.reason_prompt, role="system"), hint_msg
            )

            # Generate and parse the response
            try:
                if self.count_token:
                    n_tokens = count_openai_token(prompt, model="gpt-4-0613")
                    self.speak(f" Count input token {n_tokens} ".center(70, "#"))
                res = self.model(
                    prompt,
                    parse_func=self.reason_parser.parse,
                    max_retries=1,
                )

                end_mem_idx = self.memory.size()
                if end_mem_idx != start_mem_idx:
                    # delete the memory from the last iteration (wrong parsing)
                    delete_idx = list(range(start_mem_idx, end_mem_idx))
                    self.memory.delete(delete_idx)

                # Record the response in memory
                msg_response = Msg(self.name, res.text, "assistant")
                self.memory.add(msg_response)

                # Print out the response
                self.speak(msg_response)
                # Break the loop if the response is parsed successfully
                return res
            except ResponseParsingError as e:
                # Print out raw response from models for developers to debug
                response_msg = Msg(self.name, e.raw_response, "assistant")
                self.speak(response_msg)

                if "context_length_exceeded" in e.raw_response:
                    print(f"tackling context_length_exceeded error: {e.raw_response}")
                    error_msg = Msg(
                        "system",
                        "The response is too long. Merge similar attributes or discard irrelevant/low-confidence attributes for next check.",
                        "system",
                    )
                    # clear the memory
                    delete_idx = list(range(2, self.memory.size()))
                    self.memory.delete(delete_idx)
                    self.memory.add([error_msg])

                # Re-correct by model itself
                error_msg = Msg("system", str(e), "system")
                self.speak(error_msg)

                self.memory.add([response_msg, error_msg])

    def naive_infer(self, missing_pii: list, user_history: str) -> dict:
        self.memory.clear()
        # Put sys prompt into memory
        map_attr = SafeDict(
            attributes=", ".join(missing_pii),
            user_histories="\n".join(user_history),
        )

        naive_prompt = PROFILER_NAIVE_PROMPT.format_map(map_attr)

        # Prepare hint to remind model what the response format is
        # Won't be recorded in memory to save tokens
        hint_msg = Msg(
            "system",
            self.reason_parser.format_instruction,
            role="system",
        )

        while True:
            # Prepare prompt for the model
            prompt = self.model.format(self.memory.get_memory(), Msg("system", naive_prompt, role="system"), hint_msg)

            self.speak(f" [PROFILER] NAIVE REASON ".center(70, "#"))
            # Generate and parse the response
            try:
                if self.count_token:
                    n_tokens = count_openai_token(prompt, model="gpt-4-0613")
                    self.speak(f" Count input token {n_tokens} ".center(70, "#"))
                res = self.model(
                    prompt,
                    parse_func=self.reason_parser.parse,
                    max_retries=1,
                )

                # Record the response in memory
                msg_response = Msg(self.name, res.text, "assistant")

                # Print out the response
                self.speak(msg_response)
                # Break the loop if the response is parsed successfully
                return res
            except ResponseParsingError as e:
                # Print out raw response from models for developers to debug
                response_msg = Msg(self.name, e.raw_response, "assistant")
                self.speak(response_msg)

                if "context_length_exceeded" in e.raw_response:
                    print(f"tackling context_length_exceeded error: {e.raw_response}")
                    error_msg = Msg(
                        "system",
                        "The response is too long. Only provide necessary information.",
                        "system",
                    )
                    # clear the memory
                    delete_idx = list(range(2, self.memory.size()))
                    self.memory.delete(delete_idx)
                    self.memory.add([error_msg])

                # Re-correct by model itself
                error_msg = Msg("system", str(e), "system")
                self.speak(error_msg)

                self.memory.add([response_msg, error_msg])
