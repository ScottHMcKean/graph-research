"""LLM-based entity and relationship extraction.

Domain-agnostic: a single prompt asks the model to pull entities and the
relationships between them out of each document chunk and return strict JSON.
Works on any corpus, not just government documents. Uses ``ChatDatabricks`` so
it runs against a Databricks Foundation Model endpoint.
"""

import json
import logging
import re
from typing import List, Tuple

import pandas as pd

from .models import Document, Entity, Relationship

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "databricks-claude-sonnet-4"

SYSTEM_PROMPT = """You are an information-extraction engine that builds knowledge graphs.
From the document text, extract the salient entities and the relationships between them.

Rules:
- An entity is a named thing: a person, organization, place, product, program, event, or concept.
- entity "type" is a short UPPERCASE label, e.g. PERSON, ORGANIZATION, LOCATION, PRODUCT, PROGRAM, EVENT, CONCEPT.
- A relationship connects two entities you also listed, with a short UPPERCASE "type" verb, e.g. WORKS_FOR, LOCATED_IN, PART_OF, FOUNDED, PARTNERS_WITH, MANAGES, PRODUCES.
- "context" is a short phrase quoting/paraphrasing the supporting text.
- Only include relationships where both entities appear in your entities list.
- Return STRICT JSON only, no prose, matching this schema:
{"entities": [{"name": "...", "type": "...", "description": "..."}],
 "relationships": [{"source": "...", "target": "...", "type": "...", "context": "..."}]}"""

USER_TEMPLATE = "Document title: {title}\n\nText:\n{chunk}\n\nReturn the JSON now."


def chunk_text(text: str, max_chars: int = 6000, overlap: int = 200) -> List[str]:
    """Split text into character-bounded chunks on paragraph boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap else ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _parse_json(raw: str) -> dict:
    """Best-effort parse of model output that may be wrapped in code fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
    # Fall back to the first {...} block if there is leading/trailing prose.
    if not raw.startswith("{"):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
    return json.loads(raw)


def _build_llm(model: str, temperature: float):
    from databricks_langchain import ChatDatabricks

    return ChatDatabricks(endpoint=model, temperature=temperature, max_tokens=4000)


def extract_from_documents(
    documents: List[Document],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_chars: int = 6000,
) -> Tuple[List[Entity], List[Relationship], List[Tuple[str, str]]]:
    """Extract entities and relationships from every document via the LLM.

    Returns ``(entities, relationships, mentions)`` where ``mentions`` is a list
    of ``(doc_id, entity_name)`` pairs feeding the document->entity edges.
    """
    llm = _build_llm(model, temperature)

    entities: dict[str, Entity] = {}
    relationships: List[Relationship] = []
    rel_seen: set[tuple] = set()
    mentions: set[Tuple[str, str]] = set()

    for doc in documents:
        chunks = chunk_text(doc.content, max_chars=max_chars)
        logger.info("Extracting from %s (%d chunk(s))", doc.doc_id, len(chunks))

        for i, chunk in enumerate(chunks):
            messages = [
                ("system", SYSTEM_PROMPT),
                ("human", USER_TEMPLATE.format(title=doc.title, chunk=chunk)),
            ]
            try:
                response = llm.invoke(messages)
                payload = _parse_json(response.content)
            except Exception as exc:  # noqa: BLE001 - tolerate bad chunks
                logger.warning("Extraction failed for %s chunk %d: %s", doc.doc_id, i, exc)
                continue

            for e in payload.get("entities", []):
                name = str(e.get("name", "")).strip()
                if not name:
                    continue
                etype = str(e.get("type", "ENTITY")).strip().upper() or "ENTITY"
                desc = str(e.get("description", "")).strip()
                if name not in entities:
                    entities[name] = Entity(name=name, type=etype, description=desc)
                elif desc and not entities[name].description:
                    entities[name].description = desc
                mentions.add((doc.doc_id, name))

            for r in payload.get("relationships", []):
                src = str(r.get("source", "")).strip()
                tgt = str(r.get("target", "")).strip()
                if not src or not tgt:
                    continue
                rtype = str(r.get("type", "RELATES")).strip().upper() or "RELATES"
                key = (src, tgt, rtype)
                if key in rel_seen:
                    continue
                rel_seen.add(key)
                relationships.append(
                    Relationship(
                        source=src,
                        target=tgt,
                        type=rtype,
                        context=str(r.get("context", "")).strip()[:300],
                    )
                )

    # Keep only relationships whose endpoints we actually extracted as entities.
    valid = set(entities)
    relationships = [r for r in relationships if r.source in valid and r.target in valid]
    mention_list = sorted(mentions)

    logger.info(
        "Extracted %d entities, %d relationships, %d mentions",
        len(entities),
        len(relationships),
        len(mention_list),
    )
    return list(entities.values()), relationships, mention_list


def save_extraction(
    entities: List[Entity],
    relationships: List[Relationship],
    mentions: List[Tuple[str, str]],
    entities_path: str,
    relationships_path: str,
    mentions_path: str,
) -> None:
    """Persist extraction results to parquet."""
    pd.DataFrame([e.__dict__ for e in entities]).to_parquet(entities_path, index=False)
    pd.DataFrame([r.__dict__ for r in relationships]).to_parquet(
        relationships_path, index=False
    )
    pd.DataFrame(mentions, columns=["doc_id", "entity_name"]).to_parquet(
        mentions_path, index=False
    )
    logger.info("Wrote %s, %s, %s", entities_path, relationships_path, mentions_path)


def load_extraction(
    entities_path: str, relationships_path: str, mentions_path: str
) -> Tuple[List[Entity], List[Relationship], List[Tuple[str, str]]]:
    """Load extraction results back from parquet."""
    e_df = pd.read_parquet(entities_path)
    r_df = pd.read_parquet(relationships_path)
    m_df = pd.read_parquet(mentions_path)
    entities = [Entity(**row) for row in e_df.to_dict(orient="records")]
    relationships = [Relationship(**row) for row in r_df.to_dict(orient="records")]
    mentions = [(row["doc_id"], row["entity_name"]) for _, row in m_df.iterrows()]
    return entities, relationships, mentions
