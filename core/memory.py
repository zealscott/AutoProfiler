from typing import List, Union
from core.message import Msg


class Memory:
    """Simple conversation memory with the same interface as AgentScope's memory."""

    def __init__(self) -> None:
        self._messages: List[Msg] = []

    def add(self, msg_or_list: Union[Msg, List[Msg]]) -> None:
        """Add one or more messages to memory."""
        if isinstance(msg_or_list, list):
            self._messages.extend(msg_or_list)
        else:
            self._messages.append(msg_or_list)

    def get_memory(self) -> List[Msg]:
        """Return all stored messages."""
        return list(self._messages)

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

    def size(self) -> int:
        """Return the number of stored messages."""
        return len(self._messages)

    def delete(self, indices: Union[int, List[int]]) -> None:
        """Delete messages at the given indices."""
        if isinstance(indices, int):
            indices = [indices]
        # Delete in reverse order to preserve index validity
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self._messages):
                del self._messages[idx]
