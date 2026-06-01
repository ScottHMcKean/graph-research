"""Build a Kuzu knowledge graph from documents, entities, and relationships.

Schema (clean and generic):

    Document(doc_id, title, source)
    Entity(name, type, description)
    Document -[MENTIONS]-> Entity   (context)
    Entity   -[RELATES]->  Entity   (type, context)

A document MENTIONS every entity extracted from it; entity-to-entity edges come
from the extraction step.
"""

import logging
from pathlib import Path
from typing import List

import kuzu

from .models import Document, Entity, Relationship

logger = logging.getLogger(__name__)


def _create_schema(conn: kuzu.Connection) -> None:
    conn.execute("CREATE NODE TABLE IF NOT EXISTS Document(doc_id STRING, title STRING, source STRING, PRIMARY KEY(doc_id))")
    conn.execute("CREATE NODE TABLE IF NOT EXISTS Entity(name STRING, type STRING, description STRING, PRIMARY KEY(name))")
    conn.execute("CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Document TO Entity, context STRING)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS RELATES(FROM Entity TO Entity, type STRING, context STRING)")


def build_graph(
    db_path: str | Path,
    documents: List[Document],
    entities: List[Entity],
    relationships: List[Relationship],
    mentions: List[tuple[str, str]] | None = None,
) -> None:
    """Create a fresh Kuzu database and load all nodes and edges.

    ``mentions`` is a list of ``(doc_id, entity_name)`` pairs used to create
    Document-[MENTIONS]->Entity edges. Entity-to-entity RELATES edges are always
    created from ``relationships``.
    """
    db_path = Path(db_path)
    if db_path.exists():
        import shutil

        shutil.rmtree(db_path) if db_path.is_dir() else db_path.unlink()
        logger.info("Removed existing graph at %s", db_path)

    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)
    _create_schema(conn)

    for doc in documents:
        conn.execute(
            "CREATE (d:Document {doc_id: $doc_id, title: $title, source: $source})",
            {"doc_id": doc.doc_id, "title": doc.title, "source": doc.source},
        )
    logger.info("Inserted %d documents", len(documents))

    for ent in entities:
        conn.execute(
            "CREATE (e:Entity {name: $name, type: $type, description: $description})",
            {"name": ent.name, "type": ent.type, "description": ent.description},
        )
    logger.info("Inserted %d entities", len(entities))

    rel_count = 0
    for rel in relationships:
        try:
            conn.execute(
                """
                MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                CREATE (a)-[r:RELATES {type: $type, context: $context}]->(b)
                """,
                {"source": rel.source, "target": rel.target, "type": rel.type, "context": rel.context},
            )
            rel_count += 1
        except Exception:  # noqa: BLE001 - skip dangling endpoints
            continue
    logger.info("Inserted %d RELATES edges", rel_count)

    mention_count = 0
    valid_entities = {e.name for e in entities}
    valid_docs = {d.doc_id for d in documents}
    for doc_id, name in mentions or []:
        if name not in valid_entities or doc_id not in valid_docs:
            continue
        try:
            conn.execute(
                """
                MATCH (d:Document {doc_id: $doc_id}), (e:Entity {name: $name})
                CREATE (d)-[m:MENTIONS {context: ''}]->(e)
                """,
                {"doc_id": doc_id, "name": name},
            )
            mention_count += 1
        except Exception:  # noqa: BLE001
            continue
    logger.info("Inserted %d MENTIONS edges", mention_count)

    conn.close()
    db.close()
    logger.info("Graph built at %s", db_path)
