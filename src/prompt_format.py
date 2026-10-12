"""The prompt format: one system prompt that asks for a JSON array of calls, and the parser for the replies.

Free and constrained mode use the same prompt, so the constraint is the only difference between the two
modes (notes/decisions.md, 2026-10-12). The wording follows BFCL's default prompt; only the output format
is changed from Python call syntax to JSON.
"""

import json
import re

FORMAT_VERSION = "json-array-v1"  # saved with every run; change it whenever the prompt or parser changes

SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function calls to achieve the purpose.

Reply with only a JSON array of function calls, in this format:
[{{"name": "func_name1", "arguments": {{"param1": value1, "param2": value2}}}}, {{"name": "func_name2", "arguments": {{"param": value}}}}]
If none of the functions can be used, reply with an empty array: []
Do not include any other text in the reply.

Here is a list of functions in JSON format that you can invoke.
{functions}
"""

# One Markdown code fence around the whole reply, e.g. ```json\n[...]\n```
_FENCE = re.compile(r"^```(?:json)?[ \t]*\n?(.*?)\n?[ \t]*```$", re.DOTALL)


class ParseError(ValueError):
    """The reply is not a JSON array of calls in the requested format."""


def build_messages(question):
    """Chat messages for one question: our system prompt followed by the BFCL question."""
    system = SYSTEM_PROMPT.format(functions=json.dumps(question["functions"], indent=4))
    messages = [dict(m) for m in question["messages"]]
    # Same rule as BFCL: if the question brings its own system message, ours goes in front of it
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = system + "\n\n" + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": system})
    return messages


def parse_calls(text):
    """Parse a reply into a list of {"name": str, "arguments": dict}. Raises ParseError.

    Only surrounding whitespace and one Markdown code fence are tolerated. Nothing else is repaired,
    so any reply that would need repair counts as a syntactic failure.
    """
    s = text.strip()
    fence = _FENCE.match(s)
    if fence:
        s = fence.group(1).strip()
    try:
        calls = json.loads(s)
    except json.JSONDecodeError as e:
        raise ParseError(f"not valid JSON ({e.msg} at character {e.pos})") from None
    if not isinstance(calls, list):
        raise ParseError(f"expected a JSON array, got a JSON {type(calls).__name__}")
    for i, call in enumerate(calls):
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ParseError(f"call {i} is not an object with exactly the keys 'name' and 'arguments'")
        if not isinstance(call["name"], str):
            raise ParseError(f"call {i}: 'name' is not a string")
        if not isinstance(call["arguments"], dict):
            raise ParseError(f"call {i}: 'arguments' is not an object")
    return calls


def to_bfcl(calls):
    """Convert parsed calls to the form BFCL's checker expects: [{func_name: {param: value}}, ...]."""
    return [{call["name"]: call["arguments"]} for call in calls]
