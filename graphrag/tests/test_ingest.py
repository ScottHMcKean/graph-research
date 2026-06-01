from src.ingest import ingest_documents, load_documents, save_documents


def test_ingest_reads_text_and_markdown(tmp_path):
    (tmp_path / "a.md").write_text("# Alpha Title\n\nAlpha body text.")
    (tmp_path / "b.txt").write_text("plain body")

    docs = ingest_documents(tmp_path)

    assert len(docs) == 2
    by_id = {d.doc_id: d for d in docs}
    assert by_id["a"].title == "Alpha Title"
    assert by_id["b"].title == "B"  # filename stem, titlecased
    assert by_id["b"].content == "plain body"


def test_ingest_skips_unsupported_and_empty(tmp_path):
    (tmp_path / "good.txt").write_text("content")
    (tmp_path / "skip.json").write_text("{}")
    (tmp_path / "empty.md").write_text("   ")

    docs = ingest_documents(tmp_path)

    assert [d.doc_id for d in docs] == ["good"]


def test_save_and_load_roundtrip(tmp_path):
    (tmp_path / "x.txt").write_text("hello")
    docs = ingest_documents(tmp_path)
    out = save_documents(docs, tmp_path / "build" / "documents.parquet")
    reloaded = load_documents(out)
    assert reloaded == docs
