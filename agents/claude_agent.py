"""
Claude does not allow the system prompt in content and needs to iteratively chat with user and assistant
"""

from typing import Union, List, Sequence

import agentscope
from agentscope.message import Msg, MessageBase
from agentscope.models import load_model_by_config_name, ModelResponse, PostAPIModelWrapperBase
from agentscope.models.post_model import _convert_to_str


class ClaudePostAPIChatWrapper(PostAPIModelWrapperBase):
    """A post api model wrapper compatible with openai chat, e.g., vLLM,
    FastChat."""

    model_type: str = "claude_post_api_chat"

    def _parse_response(self, response: dict) -> ModelResponse:
        return ModelResponse(
            text=response["content"][0]["text"],
        )

    def format(
        self,
        *args: Union[MessageBase, Sequence[MessageBase]],
    ) -> Union[List[dict]]:
        """Format the input messages into a list of dict, which is
        compatible to OpenAI Chat API.

        Args:
            args (`Union[MessageBase, Sequence[MessageBase]]`):
                The input arguments to be formatted, where each argument
                should be a `Msg` object, or a list of `Msg` objects.
                In distribution, placeholder is also allowed.

        Returns:
            `Union[List[dict]]`:
                The formatted messages.
        """
        # Parse all information into a list of messages
        input_msgs = []
        for _ in args:
            if _ is None:
                continue
            if isinstance(_, MessageBase):
                input_msgs.append(_)
            elif isinstance(_, list) and all(isinstance(__, MessageBase) for __ in _):
                input_msgs.extend(_)
            else:
                raise TypeError(
                    f"The input should be a Msg object or a list " f"of Msg objects, got {type(_)}.",
                )

        messages = []

        # record dialog history as a list of strings
        dialogue = []
        for i, unit in enumerate(input_msgs):
            # Merge all messages into a dialogue history prompt
            dialogue.append(
                f"{unit.name}: {_convert_to_str(unit.content)}",
            )

        dialogue_history = "\n".join(dialogue)

        user_content_template = "## Dialogue History\n{dialogue_history}"

        messages.append(
            {
                "role": "user",
                "content": user_content_template.format(
                    dialogue_history=dialogue_history,
                ),
            },
        )

        return messages
