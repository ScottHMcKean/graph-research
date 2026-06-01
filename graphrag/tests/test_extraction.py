from src.extraction import _parse_json, chunk_text


def test_chunk_text_short_returns_single():
    assert chunk_text("short text", max_chars=100) == ["short text"]


def test_chunk_text_splits_on_paragraphs():
    text = "\n\n".join(f"paragraph {i} " + "x" * 50 for i in range(10))
    chunks = chunk_text(text, max_chars=200, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # allows for boundary slack


def test_parse_json_plain():
    raw = '{"entities": [{"name": "A", "type": "PERSON"}], "relationships": []}'
    assert _parse_json(raw)["entities"][0]["name"] == "A"


def test_parse_json_code_fenced():
    raw = '```json\n{"entities": [], "relationships": []}\n```'
    assert _parse_json(raw) == {"entities": [], "relationships": []}


def test_parse_json_with_surrounding_prose():
    raw = 'Here is the result:\n{"entities": [], "relationships": []}\nDone.'
    assert _parse_json(raw) == {"entities": [], "relationships": []}
