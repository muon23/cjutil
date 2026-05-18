from __future__ import annotations

import json
import re
from collections.abc import Generator, Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO, cast

from .TextStream import TextReader, peek_text, rewindable_text_reader


def strip_json_fence(text: str) -> str:
    """Remove optional markdown ```json fences from a text payload."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json_text(text: str) -> dict[str, Any]:
    """Parse JSON from LLM or file text, tolerating markdown code fences."""
    return json.loads(strip_json_fence(text))


def iter_json_array_stream(stream: TextReader) -> Iterator[Any]:
    """
    Yield top-level elements from a JSON array without loading the full file.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    eof = False

    while True:
        if not eof and not buffer:
            chunk = stream.read(65536)
            if not chunk:
                eof = True
            else:
                buffer += chunk

        buffer = buffer.lstrip()
        if not buffer:
            if eof:
                return
            continue

        if buffer[0] == "[":
            buffer = buffer[1:].lstrip()
            continue
        if buffer[0] in ",]":
            buffer = buffer[1:].lstrip()
            continue

        try:
            item, index = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            if eof:
                raise
            chunk = stream.read(65536)
            if not chunk:
                eof = True
                continue
            buffer += chunk
            continue

        yield item
        buffer = buffer[index:].lstrip()


def iter_json_objects(source: Any) -> Generator[Any, None, None]:
    """
    Yield JSON values from a path, open file, list, dict, iterable, or JSON array stream.

    For file sources, streams top-level array elements when the file begins with ``[``.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open(encoding="utf-8") as handle:
            yield from iter_json_objects(handle)
        return

    if isinstance(source, dict):
        yield source
        return

    if isinstance(source, list):
        yield from source
        return

    if hasattr(source, "read"):
        raw = cast(TextReader, source)
        text_io = rewindable_text_reader(raw)
        text_io, sample = peek_text(text_io)
        if not sample.lstrip():
            return

        if sample.lstrip().startswith("["):
            yield from iter_json_array_stream(text_io)
            return

        data = json.load(text_io)
        yield from iter_json_objects(data)
        return

    if isinstance(source, Iterable):
        yield from source
        return

    raise TypeError(f"unsupported JSON source type: {type(source)!r}")


def write_json_array_stream(
        output: TextIO | Path | str,
        records: Iterable[Any],
) -> None:
    """Write records as a JSON array, one element at a time (streaming-friendly)."""
    if isinstance(output, (str, Path)):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            write_json_array_stream(handle, records)
        return

    output.write("[\n")
    first = True
    for record in records:
        if not first:
            output.write(",\n")
        json.dump(record, output, ensure_ascii=False)
        first = False
    output.write("\n]\n")
