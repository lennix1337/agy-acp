"""Translate agy tool steps into ACP tool-call updates.

agy reports each tool invocation as a ``step_update`` with ``step_type`` ``tool``: an ACTIVE
update carries the parameters and the DONE update adds the output. ACP clients render these
as tool calls, so a long, silent turn stays observable while it runs.
"""

from __future__ import annotations

from collections.abc import Mapping

from acp.helpers import start_tool_call, update_tool_call
from acp.schema import ToolCallLocation, ToolCallProgress, ToolCallStart, ToolKind

from agy_acp.events import AgyStepUpdateEvent, AgyToolInfo

MAX_OUTPUT_CHARS = 4000

_KINDS: Mapping[str, ToolKind] = {
    "run_command": "execute",
    "command_status": "execute",
    "send_command_input": "execute",
    "view_file": "read",
    "list_dir": "read",
    "grep_search": "search",
    "find_by_name": "search",
    "codebase_search": "search",
    "write_to_file": "edit",
    "replace_file_content": "edit",
    "multi_replace_file_content": "edit",
    "read_url_content": "fetch",
    "search_web": "fetch",
}
_PATH_KEYS = ("AbsolutePath", "TargetFile", "DirectoryPath", "SearchDirectory", "SearchPath")
_TITLE_KEYS = ("CommandLine", "Query", "Pattern", "Url", "AbsolutePath", "TargetFile")


def tool_kind(name: str) -> ToolKind:
    return _KINDS.get(name, "other")


def _first_string(parameters: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = parameters.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _title(tool: AgyToolInfo) -> str:
    subject = _first_string(tool.parameters, _TITLE_KEYS)
    if subject is None:
        return tool.name
    return f"{tool.name}: {subject.splitlines()[0][:120]}"


def _locations(tool: AgyToolInfo) -> list[ToolCallLocation]:
    path = _first_string(tool.parameters, _PATH_KEYS)
    if path is None or not path.startswith("/"):
        return []
    return [ToolCallLocation(path=path)]


def _truncate(output: str) -> str:
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    return output[:MAX_OUTPUT_CHARS] + f"\n[truncated {len(output) - MAX_OUTPUT_CHARS} chars]"


def tool_call_id(event: AgyStepUpdateEvent) -> str:
    return f"{event.conversation_id}:{event.step_index}"


def start_update(event: AgyStepUpdateEvent, tool: AgyToolInfo) -> ToolCallStart:
    return start_tool_call(
        tool_call_id(event),
        _title(tool),
        kind=tool_kind(tool.name),
        status="in_progress",
        locations=_locations(tool),
        raw_input=dict(tool.parameters),
    )


def finish_update(
    event: AgyStepUpdateEvent, tool: AgyToolInfo, *, failed: bool = False
) -> ToolCallProgress:
    return update_tool_call(
        tool_call_id(event),
        status="failed" if failed else "completed",
        raw_output=_truncate(tool.output) if tool.output is not None else None,
    )
