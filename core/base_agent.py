from typing import Any, Callable, List, Optional, Union

import time

import litellm
from loguru import logger

from core.exceptions import ResponseParsingError
from core.memory import Memory
from core.message import ModelResponse, Msg


class LLMClient:
    """Thin wrapper around litellm.completion() replacing AgentScope model wrappers."""

    def __init__(self, model: str, max_retries: int = 20, **model_kwargs: Any) -> None:
        self.model = model
        self.max_retries = max_retries
        self.model_kwargs = model_kwargs

    def __call__(
        self,
        messages: List[dict],
        parse_func: Optional[Callable[[str], dict]] = None,
        max_retries: Optional[int] = None,
    ) -> ModelResponse:
        """Call the LLM and optionally parse the response.

        Args:
            messages: OpenAI-format message list.
            parse_func: Optional function to parse the response text.
            max_retries: Override instance max_retries for this call.

        Returns:
            ModelResponse with text and optional parsed dict.

        Raises:
            ResponseParsingError: If parse_func fails after retries.
        """
        retries = max_retries if max_retries is not None else self.max_retries
        max_rate_limit_retries = 10

        for attempt in range(retries):
            # Inner loop: retry only rate-limit / quota errors with backoff
            for rl_attempt in range(max_rate_limit_retries):
                try:
                    response = litellm.completion(
                        model=self.model,
                        messages=messages,
                        **self.model_kwargs,
                    )
                    break  # success, exit rate-limit retry loop
                except (litellm.RateLimitError, litellm.ServiceUnavailableError) as e:
                    label = "Rate limit" if isinstance(e, litellm.RateLimitError) else "Service unavailable"
                    if rl_attempt < max_rate_limit_retries - 1:
                        wait = max(20, min(2 ** rl_attempt, 120))
                        logger.warning(f"{label} (attempt {rl_attempt + 1}), waiting {wait}s")
                        time.sleep(wait)
                        continue
                    raise
                except Exception as e:
                    # Non-rate-limit error: let outer loop handle retries
                    if attempt == retries - 1:
                        raise
                    logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                    break  # break inner loop, let outer loop retry
            else:
                # All rate limit retries exhausted (shouldn't reach here due to raise above)
                continue

            try:
                text = response.choices[0].message.content or ""
            except (AttributeError, UnboundLocalError):
                if attempt == retries - 1:
                    raise
                continue

            parsed = None
            if parse_func is not None:
                try:
                    parsed = parse_func(text)
                except ResponseParsingError:
                    if attempt == retries - 1:
                        raise
                    continue

            return ModelResponse(text=text, parsed=parsed)

        # Should not reach here, but just in case
        raise RuntimeError("Exhausted retries without success or exception")

    def format(self, *args: Union[Msg, List[Msg], dict]) -> List[dict]:
        """Flatten Msg objects into an OpenAI-compatible message list.

        For Gemini models, ensures the last non-system message has role 'user'
        (Gemini requires conversations to end with a user turn).
        """
        messages = []
        for arg in args:
            if isinstance(arg, Msg):
                messages.append(arg.to_dict())
            elif isinstance(arg, list):
                for item in arg:
                    if isinstance(item, Msg):
                        messages.append(item.to_dict())
                    elif isinstance(item, dict):
                        messages.append(item)
            elif isinstance(arg, dict):
                messages.append(arg)

        # Gemini requires the last message to have role "user"
        if "gemini" in self.model.lower() and messages:
            if messages[-1]["role"] != "user":
                messages[-1] = {**messages[-1], "role": "user"}

        return messages


class AgentBase:
    """Lightweight agent base class replacing AgentScope's AgentBase."""

    def __init__(
        self,
        name: str,
        sys_prompt: str = "",
        model: str = "gpt-4o",
        **model_kwargs: Any,
    ) -> None:
        self.name = name
        self.sys_prompt = sys_prompt
        self.model = LLMClient(model=model, **model_kwargs)
        self.memory = Memory()

    def speak(self, msg: Any) -> None:
        """Print a message (replaces AgentScope's speak)."""
        if isinstance(msg, str):
            logger.info(msg)
        elif isinstance(msg, Msg):
            logger.info(f"[{msg.name}]: {msg.content}")
        else:
            logger.info(str(msg))

    def __call__(self, x: Any = None) -> Any:
        """Make the agent callable (delegates to reply)."""
        return self.reply(x)

    def reply(self, x: Any = None) -> Any:
        """Override in subclass."""
        raise NotImplementedError
