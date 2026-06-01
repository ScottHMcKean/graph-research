"""Ingest local documents into the GraphRAG pipeline.

Replaces the old web scraper. Reads a folder of plain-text, Markdown, or PDF
files and normalises them into a list of ``Document`` records (and a parquet
file for downstream steps). Drop your own corpus into ``data/docs/`` and run
``python main.py ingest``.
"""

import logging
import re
from pathlib import Path
from typing import List

import pandas as pd

from .models import Document

logger = logging.getLogger(__name__)

# File extensions we know how to read.
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


def _slugify(value: str) -> str:
    """Turn a filename stem into a stable, readable doc id."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "doc"


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfplumber (lazy import)."""
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Reading PDFs requires pdfplumber. Install it with `uv sync` or "
            "`pip install pdfplumber`."
        ) from exc

    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages).strip()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _title_from(path: Path, content: str) -> str:
    """Use the first Markdown heading if present, else the filename stem."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if line:
            break
    return path.stem.replace("_", " ").replace("-", " ").title()


def ingest_documents(docs_dir: str | Path) -> List[Document]:
    """Load every supported file under ``docs_dir`` into ``Document`` records."""
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {docs_dir}. Create it and add "
            f"some {', '.join(sorted(SUPPORTED_SUFFIXES))} files."
        )

    documents: List[Document] = []
    seen_ids: set[str] = set()

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        try:
            if path.suffix.lower() in PDF_SUFFIXES:
                content = _read_pdf(path)
            else:
                content = _read_text(path)
        except Exception as exc:  # noqa: BLE001 - skip unreadable files
            logger.warning("Skipping %s: %s", path, exc)
            continue

        if not content:
            logger.warning("Skipping empty document: %s", path)
            continue

        # Make doc ids unique even if stems collide across subfolders.
        doc_id = _slugify(path.stem)
        suffix_n = 1
        while doc_id in seen_ids:
            suffix_n += 1
            doc_id = f"{_slugify(path.stem)}_{suffix_n}"
        seen_ids.add(doc_id)

        documents.append(
            Document(
                doc_id=doc_id,
                title=_title_from(path, content),
                source=str(path.relative_to(docs_dir)),
                content=content,
            )
        )
        logger.info("Ingested %s (%d chars)", path.name, len(content))

    logger.info("Ingested %d documents from %s", len(documents), docs_dir)
    return documents


def save_documents(documents: List[Document], out_path: str | Path) -> Path:
    """Persist documents to parquet for the extraction step."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([d.__dict__ for d in documents])
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d documents -> %s", len(documents), out_path)
    return out_path


def load_documents(path: str | Path) -> List[Document]:
    """Load documents back from parquet."""
    df = pd.read_parquet(path)
    return [Document(**row) for row in df.to_dict(orient="records")]
