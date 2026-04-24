import html
import json
import re
from typing import Optional

from inspect_ai.model._chat_message import ChatMessageAssistant
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleHandler
from inspect_ai.model._providers.util.llama31 import (
    filter_assistant_header,
    parse_tool_call_content,
)
from inspect_ai.tool._tool_info import ToolInfo

_RECOVERABLE_TAGS = ("tool_call", "tools", "function")
_PATCH_ATTR = "_reasoning_blind_spots_patch_applied"


def _extract_leading_json_object(text: str) -> Optional[tuple[str, str]]:
    """
    Extract the first balanced JSON object from the start of a string.

    This is tolerant of an omitted closing `</tool_call>` tag, which some
    OpenAI-compatible models emit when using Inspect AI's tool emulation prompt.
    """

    # Ignore leading whitespace before the JSON object.
    match = re.search(r"\{", text)
    if match is None:
        return None

    start = match.start()
    depth = 0
    in_string = False
    escaped = False

    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1], text[idx + 1 :]

    return None


def _extract_balanced_json_from_index(text: str, start: int) -> Optional[str]:
    depth = 0
    in_string = False
    escaped = False

    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def _normalize_tool_payload(payload: str) -> str:
    candidate = html.unescape(payload).strip()
    json.loads(candidate)
    return candidate


def _repair_tool_payload(payload: str) -> Optional[str]:
    candidate = html.unescape(payload).strip()

    try:
        return _normalize_tool_payload(candidate)
    except Exception:
        pass

    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', candidate, flags=re.DOTALL)
    args_match = re.search(
        r'"(arguments|parameters)"\s*:\s*([\{\[])', candidate, flags=re.DOTALL
    )
    if not name_match or not args_match:
        return None

    name = name_match.group(1)
    args_start = args_match.start(2)
    args_json = _extract_balanced_json_from_index(candidate, args_start)
    if args_json is None:
        return None

    try:
        arguments = json.loads(args_json)
    except Exception:
        return None

    repaired = json.dumps({"name": name, "arguments": arguments})
    try:
        return _normalize_tool_payload(repaired)
    except Exception:
        return None


def _extract_closed_tag_payloads(
    response: str, tag: str
) -> tuple[list[str], list[str]] | None:
    tag_regex = rf"<{tag}>((?:.|\n)*?)</{tag}>"
    payloads = re.findall(tag_regex, response, flags=re.DOTALL)
    if not payloads:
        return None

    other_content = re.split(rf"<{tag}>(?:.|\n)*?</{tag}>", response, flags=re.DOTALL)
    content_parts = [part.strip() for part in other_content if part.strip()]
    return payloads, content_parts


def _extract_open_tag_payloads(
    response: str, tag: str
) -> tuple[list[str], list[str]] | None:
    open_tag = f"<{tag}>"
    if open_tag not in response:
        return None

    segments = response.split(open_tag)
    content_parts = [segments[0].strip()] if segments[0].strip() else []
    payloads = []

    for segment in segments[1:]:
        stripped = segment.lstrip()
        if not stripped:
            continue

        extracted = _extract_leading_json_object(stripped)
        if extracted is None:
            raw_payload = stripped.split("<", 1)[0].strip()
            if raw_payload.startswith("{"):
                payloads.append(raw_payload)
            else:
                trailing = stripped.strip()
                if trailing:
                    content_parts.append(f"{open_tag}\n{trailing}")
            continue

        payload, trailing = extracted
        payloads.append(payload)

        trailing_text = trailing.strip()
        if trailing_text:
            content_parts.append(trailing_text)

    if not payloads:
        return None

    return payloads, content_parts


def _recover_unclosed_tool_calls(
    response: str, tools: list[ToolInfo], model: str
) -> Optional[ChatMessageAssistant]:
    for tag in _RECOVERABLE_TAGS:
        extracted = _extract_closed_tag_payloads(response, tag)
        if extracted is None:
            extracted = _extract_open_tag_payloads(response, tag)

        if extracted is None:
            continue

        payloads, content_parts = extracted
        normalized_payloads = []
        for payload in payloads:
            repaired = _repair_tool_payload(payload)
            normalized_payloads.append(repaired or html.unescape(payload).strip())

        tool_calls = [
            parse_tool_call_content(normalized_payload, tools)
            for normalized_payload in normalized_payloads
        ]

        if not tool_calls:
            continue

        content = "\n\n".join(part for part in content_parts if part).strip()
        return ChatMessageAssistant(
            content=filter_assistant_header(content),
            tool_calls=tool_calls,
            model=model,
            source="generate",
        )

    return None


def patch_inspect_tool_emulation() -> None:
    """
    Patch Inspect AI's OpenAI-compatible tool emulation to recover unclosed
    `<tool_call>` blocks emitted by some externally hosted models.
    """

    if getattr(OpenAICompatibleHandler, _PATCH_ATTR, False):
        return

    original_parse = OpenAICompatibleHandler.parse_assistant_response

    def patched_parse_assistant_response(
        self, response: str, tools: list[ToolInfo]
    ) -> ChatMessageAssistant:
        message = original_parse(self, response, tools)
        if message.tool_calls:
            return message

        recovered = _recover_unclosed_tool_calls(response, tools, self.model)
        if recovered is not None:
            return recovered

        return message

    OpenAICompatibleHandler.parse_assistant_response = patched_parse_assistant_response
    setattr(OpenAICompatibleHandler, _PATCH_ATTR, True)
