"""Utilities for extracting JSON payloads from model text."""


def extract_json_object(text: str) -> str:
    """Extract the first JSON object from text or return the original text."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].lstrip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped
