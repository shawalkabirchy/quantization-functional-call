"""JSON Schema for a reply: a JSON array of calls to the given functions.

Constrained mode passes this schema to vLLM's guided decoding (xgrammar backend), so the model can only
write replies that match it. This is the first version (Update 3); Update 7 reuses it to separate
syntactic failures (reply does not match the schema) from semantic ones.

BFCL function definitions use their own type names, which are mapped to JSON Schema here. Only type,
enum, items, properties and required are kept; descriptions, defaults and the few format/length limits
are dropped (xgrammar in vLLM 0.8.5 does not support length limits or string formats).
"""

TYPE_MAP = {
    "string": "string",
    "integer": "integer",
    "float": "number",
    "boolean": "boolean",
    "array": "array",
    "tuple": "array",
    "dict": "object",
}


def _matches_type(value, json_type):
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    return False


def value_schema(spec):
    """Schema for one parameter (or one array item) of a BFCL function definition."""
    json_type = TYPE_MAP.get(spec.get("type"))
    if json_type is None:  # "any", or a type name BFCL does not define: no constraint
        return {}
    schema = {"type": json_type}
    # Keep an enum only if every value has the declared type; otherwise no reply could match it
    if "enum" in spec and all(_matches_type(v, json_type) for v in spec["enum"]):
        schema["enum"] = list(spec["enum"])
    if json_type == "array":
        # Set explicitly: xgrammar's strict mode would otherwise allow only an empty array
        schema["items"] = value_schema(spec["items"]) if "items" in spec else {}
    if json_type == "object":
        if "properties" in spec:
            schema["properties"] = {k: value_schema(v) for k, v in spec["properties"].items()}
            schema["required"] = [k for k in spec.get("required", []) if k in spec["properties"]]
            # Extra keys are not allowed (BFCL's checker rejects unexpected parameters too)
            schema["additionalProperties"] = False
        else:
            # A free-form dict: any keys
            schema["additionalProperties"] = True
    return schema


def call_schema(function):
    """Schema for one call: {"name": <this function's name>, "arguments": {...}}."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": [function["name"]]},
            "arguments": value_schema(function["parameters"]),
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }


def calls_schema(functions):
    """Schema for a whole reply: an array (possibly empty) of calls to any of the given functions."""
    variants = [call_schema(f) for f in functions]
    return {"type": "array", "items": variants[0] if len(variants) == 1 else {"anyOf": variants}}
