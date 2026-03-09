import json
import re
from typing import Any, Dict, List, Optional

import dirtyjson

from core.exceptions import ResponseParsingError


class MarkdownJsonDictParser:
    """Parses JSON from markdown code blocks, compatible with AgentScope's interface."""

    def __init__(
        self,
        content_hint: Optional[Dict[str, Any]] = None,
        required_keys: Optional[List[str]] = None,
    ) -> None:
        self.content_hint = content_hint or {}
        self.required_keys = required_keys or []

    @property
    def format_instruction(self) -> str:
        """Generate a format hint string for the LLM."""
        hint = json.dumps(self.content_hint, indent=4, ensure_ascii=False)
        return (
            "You should respond a JSON object in a markdown JSON code block as follows:\n"
            "```json\n"
            f"{hint}\n"
            "```"
        )

    def parse(self, text: str) -> dict:
        """Extract and parse JSON from a markdown code block.

        Falls back to dirtyjson for malformed JSON.
        """
        # Try to extract from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        raw = match.group(1).strip() if match else text.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            try:
                result = dirtyjson.loads(raw)
                # dirtyjson returns AttributedDict; convert to plain dict
                result = json.loads(json.dumps(result))
            except Exception:
                raise ResponseParsingError(
                    f"Failed to parse JSON from response:\n{raw}",
                    raw_response=text,
                )

        if not isinstance(result, dict):
            raise ResponseParsingError(
                f"Expected a JSON dict, got {type(result).__name__}",
                raw_response=text,
            )

        # Validate required keys
        missing = [k for k in self.required_keys if k not in result]
        if missing:
            raise ResponseParsingError(
                f"Missing required keys: {missing}",
                raw_response=text,
            )

        return result
