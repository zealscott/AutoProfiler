# -*- coding: utf-8 -*-
from typing import Any

from loguru import logger
import json

from agentscope.exception import ResponseParsingError
from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.parsers import MarkdownJsonDictParser

from util.prompt_loader import SUMMARIZER_CHECK_PROMPT, SUMMARIZER_SUMMARY_PROMPT, attr_converter
from util.data_loader import SafeDict

from agentscope.utils.token_utils import count_openai_token


class Summarizer(AgentBase):
    """An agent class that used to summarize, reflect the inferred personal information.

    Note that the `Summarizer` agent is only used for reasoning. It do not store any data and do not have any tools.
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

        map_attr = SafeDict(
            attributes=", ".join(target_attributes),
            attributes_info=attr_converter(target_attributes, "string"),
        )

        self.check_prompt = "\n".join(
            [
                sys_prompt.format(name=self.name),
                SUMMARIZER_CHECK_PROMPT.format_map(map_attr),
            ],
        )
        self.summary_prompt = "\n".join(
            [
                sys_prompt.format(name=self.name),
                SUMMARIZER_SUMMARY_PROMPT,
            ],
        )

        self.check_parser = MarkdownJsonDictParser(
            content_hint={
                "thought": "how do you check and summarize the inferred information",
                "results": str(
                    [
                        {
                            "type": "{attribute name}",
                            "confidence": "{confidence score}",
                            "evidence": "{detailed facts for guessing}",
                            "guess": "{inferred information}",
                        }
                    ]
                ),
            },
            required_keys=["results"],
        )

        self.summary_parser = MarkdownJsonDictParser(
            content_hint={
                "think": "how do you summarize the inferred information",
                "summary": "the natual language summary of the inferred information",
            },
            required_keys=["summary"],
        )

    def reply(self, x: dict = None) -> dict:
        raise NotImplementedError

    def check(self, x: dict = None) -> dict:
        # clear the memory
        self.memory.clear()
        # add check system prompt
        self.memory.add(Msg("system", self.check_prompt, role="system"))
        # add current input
        self.memory.add(x)

        # The agent will keep parsing the response until it is parsed successfully
        while True:
            self.speak(f" [SUMMARIZER] CHECK ".center(70, "#"))

            # Prepare hint to remind model what the response format is
            # Won't be recorded in memory to save tokens
            hint_msg = Msg(
                "system",
                self.check_parser.format_instruction,
                role="system",
            )

            # Prepare prompt for the model
            prompt = self.model.format(self.memory.get_memory(), hint_msg)

            # Generate and parse the response
            try:
                if self.count_token:
                    n_tokens = count_openai_token(prompt, model="gpt-4-0613")
                    self.speak(f" Count input token {n_tokens} ".center(70, "#"))
                # print(prompt)
                res = self.model(
                    prompt,
                    parse_func=self.check_parser.parse,
                    max_retries=1,
                )

                if type(res.parsed["results"]) != list:
                    _type = type(res.parsed["results"])
                    raise ResponseParsingError(
                        f"The results should be a list of dict, not {_type}.",
                        res.text,
                    )

                # do not store the response in memory
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
                        "The attributes information is too long. Merge similar attributes or discard irrelevant/low-confidence attributes for next check. You also do not need to provide the thought.",
                        "system",
                    )
                    # clear the memory
                    delete_idx = list(range(2, self.memory.size()))
                    self.memory.delete(delete_idx)
                    self.memory.add([error_msg])

                else:
                    # normal parsing error
                    # Re-correct by model itself
                    error_msg = Msg("system", str(e), "system")
                    self.speak(error_msg)

                    self.memory.add([response_msg, error_msg])

    def summary(self, x: dict = None) -> dict:
        # clear the memory
        self.memory.clear()
        # add check system prompt
        self.memory.add(Msg("system", self.summary_prompt, role="system"))
        # add current input
        self.memory.add(x)

        # The agent will keep parsing the response until it is parsed successfully
        while True:
            self.speak(f" [SUMMARIZER] SUMMARY ".center(70, "#"))

            # Prepare hint to remind model what the response format is
            # Won't be recorded in memory to save tokens
            hint_msg = Msg(
                "system",
                self.summary_parser.format_instruction,
                role="system",
            )

            # Prepare prompt for the model
            prompt = self.model.format(
                self.memory.get_memory(), Msg("system", self.summary_prompt, role="system"), hint_msg
            )

            # Generate and parse the response
            try:
                if self.count_token:
                    n_tokens = count_openai_token(prompt, model="gpt-4-0613")
                    self.speak(f" Count input token {n_tokens} ".center(70, "#"))
                res = self.model(
                    prompt,
                    parse_func=self.summary_parser.parse,
                    max_retries=1,
                )

                # do not store the response in memory
                msg_response = Msg(self.name, res.text, "assistant")
                self.speak(msg_response)

                # Break the loop if the response is parsed successfully
                return res
            except ResponseParsingError as e:
                # Print out raw response from models for developers to debug
                response_msg = Msg(self.name, e.raw_response, "assistant")
                self.speak(response_msg)

                # Re-correct by model itself
                error_msg = Msg("system", str(e), "system")
                self.speak(error_msg)

                self.memory.add([response_msg, error_msg])
