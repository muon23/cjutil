from __future__ import annotations

from typing import Protocol


class TextReader(Protocol):
    """Minimal read-only text stream (file handle or PrefixedTextReader)."""

    def read(self, n: int = -1, /) -> str: ...


class PrefixedTextReader:
    """Prepends a string before delegating reads to an underlying stream."""

    def __init__(self, prefix: str, underlying: TextReader):
        self._prefix = prefix
        self._underlying = underlying
        self._spent = False

    def read(self, n: int = -1, /) -> str:
        if not self._spent:
            self._spent = True
            if n < 0:
                return self._prefix + self._underlying.read()
            chunk = self._prefix[:n]
            self._prefix = self._prefix[n:]
            if len(chunk) < n:
                chunk += self._underlying.read(n - len(chunk))
            return chunk
        return self._underlying.read(n)


def rewindable_text_reader(source: TextReader) -> TextReader:
    """Return a reader at the start; wrap when the stream cannot seek."""
    first = source.read(1)
    if not first:
        return source
    if hasattr(source, "seek"):
        source.seek(0)
        return source
    return PrefixedTextReader(first, source)


def peek_text(source: TextReader, size: int = 2048) -> tuple[TextReader, str]:
    """Read up to size chars for inspection, then return a reader rewound to the start."""
    sample = source.read(size)
    if hasattr(source, "seek"):
        source.seek(0)
        return source, sample
    return PrefixedTextReader(sample, source), sample
