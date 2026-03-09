from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Msg:
    """A simple message object replacing AgentScope's Msg."""

    name: str
    content: Any
    role: str = "user"

    def to_dict(self) -> dict:
        """Convert to OpenAI message format."""
        content = self.content if isinstance(self.content, str) else str(self.content)
        return {"role": self.role, "content": content}

    def __str__(self) -> str:
        return f"{self.name}: {self.content}"


@dataclass
class ModelResponse:
    """Wraps an LLM response with optional parsed output."""

    text: str
    parsed: Optional[dict] = None
