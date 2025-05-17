# -*- coding: utf-8 -*-
from typing import Any

from loguru import logger
import json

from agentscope.exception import ResponseParsingError, FunctionCallError
from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.parsers import MarkdownJsonDictParser
from agentscope.service import ServiceToolkit

from util.prompt_loader import RETRIEVER_PROMPT
from util.parsing import parsing_function_response

from agentscope.utils.token_utils import count_openai_token


class Retriever(AgentBase):
    """An agent class that used to collect external knowledge from Internet.

    Note that the `Retriever` agent is only used for collecting/summarying required information,
    and it will not perform reasoning steps.
    """

    def __init__(
        self,
        name: str,
        model_config_name: str,
        service_toolkit: ServiceToolkit = None,
        sys_prompt: str = "You're a helpful  information retriever. Your name is {name}.",
        max_iters: int = 20,
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
            service_toolkit (`ServiceToolkit`):
                A `ServiceToolkit` object that contains the tool functions.
        """
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        
        self.count_token = count_token
        self.model.max_retries = 20

        if service_toolkit is None:
            raise ValueError(
                "The argument `service_toolkit` is required to initialize " "the RETRIEVER.",
            )

        self.service_toolkit = service_toolkit
        self.max_iters = max_iters

        if not sys_prompt.endswith("\n"):
            sys_prompt = sys_prompt + "\n"

        self.sys_prompt = "\n".join(
            [
                # The brief intro of the role and target
                sys_prompt.format(name=self.name),
                # The instruction prompt for tools
                self.service_toolkit.tools_instruction,
                # The detailed instruction prompt for the agent
                RETRIEVER_PROMPT,
            ],
        )

        # Initialize a parser object to formulate the response from the model
        self.parser = MarkdownJsonDictParser(
            content_hint={
                "thought": "how to retrieve the information",
                "function": service_toolkit.tools_calling_format,
            },
            required_keys=["function"],
        )

    def reply(self, x: dict = None) -> dict:
        # clear the memory
        self.memory.clear()
        # Put sys prompt into memory
        self.memory.add(Msg("system", self.sys_prompt, role="system"))
        # add the instruction
        self.memory.add(x)

        # The agent will keep parsing the response until it is parsed successfully
        for i in range(self.max_iters):
            self.speak(f" [RETRIEVER] Iter {i+1} STEP 1: PARSING ".center(70, "#"))

            # Prepare hint to remind model what the response format is
            # Won't be recorded in memory to save tokens
            hint_msg = Msg(
                "system",
                self.parser.format_instruction,
                role="system",
            )

            # Prepare prompt for the model
            prompt = self.model.format(self.memory.get_memory(), hint_msg)

            # Generate and parse the response
            try:
                if self.count_token:
                    n_tokens = count_openai_token(prompt, model="gpt-4-0613")
                    self.speak(f" Count input token {n_tokens} ".center(70, "#"))
                res = self.model(
                    prompt,
                    parse_func=self.parser.parse,
                    max_retries=1,
                )

                # Record the response in memory
                msg_response = Msg(self.name, res.text, "assistant")
                self.memory.add(msg_response)

                # Print out the response
                self.speak(
                    Msg(self.name, res.text, "assistant"),
                )

            except ResponseParsingError as e:
                # Print out raw response from models for developers to debug
                response_msg = Msg(self.name, e.raw_response, "assistant")
                self.speak(response_msg)

                # Re-correct by model itself
                error_msg = Msg("system", str(e), "system")
                self.speak(error_msg)

                self.memory.add([response_msg, error_msg])

                # Skip acting step to re-correct the response
                continue

            # Step 2: Acting
            self.speak(f" [RETRIEVER] STEP 2: ACTING ".center(70, "#"))

            # Parse, check and execute the tool functions in service toolkit
            try:
                response_results = self.service_toolkit.parse_and_call_func(
                    res.parsed["function"],
                )
                # parsing the results
                status, results = parsing_function_response(response_results)
                if status == "fail":
                    raise FunctionCallError(
                        "The function calling failed. Need re-parsing the response.",
                    )

                ans = {"results": results}

            except Exception as e:
                # Catch the function calling error that can be handled by
                # the model
                error_msg = Msg("system", str(e), "system")
                self.speak(error_msg)
                self.memory.add(error_msg)

                continue

            res_msg = Msg(self.name, ans, "assistant")
            self.speak(res_msg)
            return res_msg

        # Exceed the maximum iterations
        ans = {
            "status": "fail",
            "thought": "I have failed to generate a response in the maximum iterations. Please provide a more detailed instruction For example, directly ask me to retrieval more user's comments history.",
            "results": "",
        }
        res_msg = Msg(self.name, ans, "assistant")
        self.speak(res_msg)

        return res_msg
