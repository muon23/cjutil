from pathlib import Path
from typing import TypeAlias

NlpInput: TypeAlias = str | Path | list[str] | list[Path]


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError(
                "pypdf is required to read PDF files. Install it with: pip install pypdf"
            ) from e
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    return path.read_text(encoding="utf-8").strip()


def _to_text(item: str | Path) -> str:
    path = Path(item)
    if path.is_file():
        return _read_file(path)
    return str(item).strip()


def to_texts(data: NlpInput) -> tuple[list[str], bool]:
    """
    Normalize caller input to a list of document strings.

    Args:
        data: One text or file path, a list of texts/paths, or Path objects.

    Returns:
        (texts, is_batch) where is_batch is True when the caller passed a list
        (including a list of one file) and results should be returned as list[list].
    """
    if isinstance(data, list):
        if not data:
            return [], True
        return [_to_text(item) for item in data], True

    return [_to_text(data)], False


def wrap_list_of_lists(rows: list[list[str]], is_batch: bool) -> list[str] | list[list[str]]:
    if is_batch:
        return rows
    return rows[0] if rows else []
