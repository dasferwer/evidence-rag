import pytest

from evidencerag.chunking import chunk_text, normalize_text


def test_normalize_text_preserves_paragraphs() -> None:
    assert normalize_text("one  two\r\n\r\n\r\nthree") == "one two\n\nthree"


def test_chunking_is_bounded_and_ordered() -> None:
    text = "\n\n".join(["Alpha sentence. " * 20, "Beta sentence. " * 20])
    chunks = chunk_text(text, max_chars=180, overlap_chars=30)

    assert len(chunks) > 2
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(0 < len(chunk.text) <= 220 for chunk in chunks)
    assert all(chunk.token_count > 0 for chunk in chunks)


def test_chunking_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_chars"):
        chunk_text("content", max_chars=100, overlap_chars=100)
