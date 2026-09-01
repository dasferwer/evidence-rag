import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    ordinal: int
    text: str
    token_count: int


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str, *, max_chars: int = 1_200, overlap_chars: int = 180) -> list[TextChunk]:
    """Split on paragraphs, then hard boundaries, preserving a small context overlap."""
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be lower than max_chars")

    normalized = normalize_text(text)
    if not normalized:
        return []

    paragraphs = normalized.split("\n\n")
    blocks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pending = paragraph
        while len(pending) > max_chars:
            split_at = pending.rfind(". ", 0, max_chars)
            if split_at < max_chars // 2:
                split_at = max_chars
            blocks.append(pending[:split_at].strip())
            pending = pending[split_at:].lstrip(". ")

        candidate = f"{current}\n\n{pending}".strip() if current else pending
        if current and len(candidate) > max_chars:
            blocks.append(current)
            current = f"{current[-overlap_chars:]}\n\n{pending}".strip()
        else:
            current = candidate

    if current:
        blocks.append(current)

    return [
        TextChunk(ordinal=index, text=block, token_count=max(1, len(block.split())))
        for index, block in enumerate(blocks)
    ]
