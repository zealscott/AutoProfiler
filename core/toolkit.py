import inspect
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional


class ServiceExecStatus(IntEnum):
    SUCCESS = 1
    ERROR = -1


@dataclass
class ServiceResponse:
    """Response from a service function."""
    status: ServiceExecStatus
    content: Any

    def __str__(self) -> str:
        return f"[STATUS]: {self.status.name}\n[RESULT]: {self.content}"


class ServiceToolkit:
    """Registry for tool functions with the same interface as AgentScope's ServiceToolkit."""

    def __init__(self) -> None:
        self._tools: Dict[str, dict] = {}

    def add(self, func: Callable, **bound_kwargs: Any) -> None:
        """Register a tool function with pre-bound keyword arguments.

        Only parameters NOT in bound_kwargs will be exposed to the LLM.
        """
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        # Determine which params to expose (exclude pre-bound ones)
        exposed_params = []
        for name, param in sig.parameters.items():
            if name in bound_kwargs:
                continue
            desc = ""
            # Try to extract param description from docstring
            # Look for "name (`type`): description" or "name: description"
            for pattern in [
                rf"{name}\s*\(`[^`]*`\)[^:]*:\s*(.+?)(?:\n\s*\w|\Z)",
                rf"{name}\s*:\s*(.+?)(?:\n\s*\w|\Z)",
            ]:
                match = inspect.re.search(pattern, doc, inspect.re.DOTALL) if hasattr(inspect, 're') else None
                if match is None:
                    import re
                    match = re.search(pattern, doc, re.DOTALL)
                if match:
                    desc = match.group(1).strip()
                    break

            exposed_params.append({
                "name": name,
                "description": desc,
                "default": None if param.default is inspect.Parameter.empty else param.default,
                "required": param.default is inspect.Parameter.empty,
            })

        # Extract the first line of the docstring as description
        func_desc = doc.split("\n")[0].strip() if doc else func.__name__

        self._tools[func.__name__] = {
            "func": func,
            "bound_kwargs": bound_kwargs,
            "params": exposed_params,
            "description": func_desc,
        }

    @property
    def tools_instruction(self) -> str:
        """Generate tool descriptions from registered functions."""
        lines = ["## Available Tools\n"]
        for name, info in self._tools.items():
            lines.append(f"### {name}")
            lines.append(f"{info['description']}\n")
            if info["params"]:
                lines.append("**Parameters:**")
                for p in info["params"]:
                    req = " (required)" if p["required"] else f" (default: {p['default']})"
                    lines.append(f"- `{p['name']}`{req}: {p['description']}")
            lines.append("")
        return "\n".join(lines)

    @property
    def tools_calling_format(self) -> str:
        """Format hint string for tool calls."""
        tool_names = list(self._tools.keys())
        return (
            "[{\"name\": \"<function_name>\", \"arguments\": {<arg_name>: <arg_value>, ...}}] "
            f"where function_name is one of {tool_names}"
        )

    def parse_and_call_func(self, func_call: Any) -> str:
        """Parse a function call specification and execute it.

        Args:
            func_call: A list/dict with 'name' and 'arguments' keys.

        Returns:
            Formatted string with [STATUS] and [RESULT].
        """
        # Normalize to list
        if isinstance(func_call, dict):
            func_call = [func_call]
        elif isinstance(func_call, str):
            try:
                func_call = json.loads(func_call)
                if isinstance(func_call, dict):
                    func_call = [func_call]
            except json.JSONDecodeError:
                return f"[STATUS]: {ServiceExecStatus.ERROR.name}\n[RESULT]: Failed to parse function call: {func_call}"

        results = []
        for call in func_call:
            name = call.get("name", "")
            args = call.get("arguments", {})

            if name not in self._tools:
                results.append(
                    f"[STATUS]: {ServiceExecStatus.ERROR.name}\n"
                    f"[RESULT]: Unknown function '{name}'"
                )
                continue

            tool = self._tools[name]
            # Merge bound kwargs with call args
            merged = {**tool["bound_kwargs"], **args}

            try:
                response = tool["func"](**merged)
                if isinstance(response, ServiceResponse):
                    results.append(str(response))
                else:
                    results.append(
                        f"[STATUS]: {ServiceExecStatus.SUCCESS.name}\n"
                        f"[RESULT]: {response}"
                    )
            except Exception as e:
                results.append(
                    f"[STATUS]: {ServiceExecStatus.ERROR.name}\n"
                    f"[RESULT]: {e}"
                )

        return "\n\n".join(results)
