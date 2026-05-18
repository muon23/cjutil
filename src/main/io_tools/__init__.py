"""Streaming text and JSON I/O utilities."""

from .JsonStream import (
    iter_json_array_stream,
    iter_json_objects,
    parse_json_text,
    strip_json_fence,
    write_json_array_stream,
)
from .TextStream import PrefixedTextReader, TextReader, peek_text, rewindable_text_reader

__all__ = [
    "PrefixedTextReader",
    "TextReader",
    "iter_json_array_stream",
    "iter_json_objects",
    "parse_json_text",
    "peek_text",
    "rewindable_text_reader",
    "strip_json_fence",
    "write_json_array_stream",
]
