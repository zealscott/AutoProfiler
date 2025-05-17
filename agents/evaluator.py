"""
Design a set of evaluators to evaluate the PII
"""

from loguru import logger
import json

from agentscope.exception import ResponseParsingError
from agentscope.agents import AgentBase
from agentscope.message import Msg

from util.prompt_loader import EVALUATOR_PROMPT
from util.parsing import parsing_function_response
from util.data_loader import SafeDict

from agentscope.utils.token_utils import count_openai_token


class Evaluator(AgentBase):
    """An agent class that used to evaluate the PII."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        sys_prompt: str = "You're a helpful evaluator. Your name is {name}.",
        count_token: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        self.count_token = count_token
        self.model.max_retries = 20

        if not sys_prompt.endswith("\n"):
            sys_prompt = sys_prompt + "\n"

        self.sys_prompt = sys_prompt.format(name=self.name)

    def reply(self, ground_truth: str, pred: str) -> dict:
        # clear the memory
        self.memory.clear()

        map_attr = SafeDict(
            ground_truth_value=ground_truth,
            prediction_value=pred,
        )

        prompt = "\n".join(
            [
                self.sys_prompt,
                EVALUATOR_PROMPT.format_map(map_attr),
            ],
        )

        # Put sys prompt into memory
        self.memory.add(Msg("system", prompt, role="system"))

        while True:
            # self.speak(f" [EVALUATOR] ".center(70, "#"))

            # Prepare prompt for the model
            prompt = self.model.format(self.memory.get_memory())

            # Generate and parse the response
            try:
                res = self.model(
                    prompt,
                    max_retries=20,
                )
                res = res.text.strip("'")
                # print(res)

                if res not in ["yes", "no", "less precise"]:
                    raise ResponseParsingError(
                        "The response should be 'yes', 'no', or 'less precise'.",
                        res,
                    )

                # do not store the response in memory
                msg_response = Msg(self.name, res, "assistant")
                # Print out the response
                # self.speak(msg_response)

                # Break the loop if the response is parsed successfully
                return res
            except ResponseParsingError as e:
                # Print out raw response from models for developers to debug
                response_msg = Msg(self.name, e.raw_response, "assistant")
                self.speak(response_msg)

                error_msg = Msg("system", str(e), "system")
                self.speak(error_msg)

                self.memory.add([response_msg, error_msg])
