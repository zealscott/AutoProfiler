from typing import Any

from loguru import logger

from core.exceptions import ResponseParsingError
from core.base_agent import AgentBase
from core.message import Msg
from core.parser import MarkdownJsonDictParser


class Tagger(AgentBase):
    """An agent that tags personal attributes from text."""

    def __init__(
        self,
        name: str,
        model: str = "gpt-4o",
        sys_prompt: str = "You're a helpful tagger. Your name is {name}.",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            **kwargs,
        )

        self.sys_prompt = sys_prompt + "\n"

        self.parser = MarkdownJsonDictParser(
            content_hint={
                "think": "what do you think about the situation",
                "result": "attribute name, separated by comma",
            },
            required_keys=["think", "result"],
        )

    def reply(self, x: Msg = None) -> Any:
        # clear all previous memory
        self.memory.clear()
        # Put sys prompt into memory
        self.memory.add(Msg("system", self.sys_prompt, role="system"))

        # store the input message
        self.memory.add(x)
        self.speak(x)

        # Keep parsing until success
        while True:
            self.speak(f" [Tagger] tagging ".center(70, "#"))

            # Prepare hint (not recorded in memory)
            hint_msg = Msg(
                "system",
                self.parser.format_instruction,
                role="system",
            )

            # Prepare prompt
            prompt = self.model.format(self.memory.get_memory(), hint_msg)

            try:
                res = self.model(
                    prompt,
                    parse_func=self.parser.parse,
                    max_retries=20,
                )

                msg_response = Msg(self.name, res.text, "assistant")
                self.speak(msg_response)
                return res

            except ResponseParsingError as e:
                response_msg = Msg(self.name, e.raw_response, "assistant")
                self.speak(response_msg)

                if "context_length_exceeded" in e.raw_response:
                    logger.warning(f"Context length exceeded: {e.raw_response}")
                    error_msg = Msg(
                        "system",
                        "The response is too long. Only provide necessary information.",
                        "system",
                    )
                    delete_idx = list(range(2, self.memory.size()))
                    self.memory.delete(delete_idx)
                    self.memory.add([error_msg])

                # Re-correct by model itself
                error_msg = Msg("system", str(e), "system")
                self.speak(error_msg)
                self.memory.add([response_msg, error_msg])
