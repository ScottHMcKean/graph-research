"""GraphRAG query tools over the Kuzu knowledge graph.

These are the LangChain tools the agent calls to retrieve from the graph.
They are domain-agnostic: they operate on the generic Document/Entity schema
(see ``graph_build.py``) rather than any particular dataset.
"""

import logging
from typing import Any, Dict, List, Optional

import kuzu
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_connection: Optional[kuzu.Connection] = None
_db_path: str = "build/graph.kuzu"


def initialize_db_connection(db_path: Optional[str] = None) -> None:
    """Open a read-only connection to the graph (idempotent)."""
    global _connection, _db_path
    if db_path:
        _db_path = db_path
    if _connection is None:
        db = kuzu.Database(_db_path, read_only=True)
        _connection = kuzu.Connection(db)


def _conn() -> kuzu.Connection:
    if _connection is None:
        initialize_db_connection()
    return _connection


def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Run a Cypher query and return a list of row dicts."""
    try:
        result = _conn().execute(query, params or {})
        columns = result.get_column_names()
        rows = []
        while result.has_next():
            rows.append(dict(zip(columns, result.get_next())))
        return rows
    except Exception as exc:  # noqa: BLE001 - tools should degrade gracefully
        logger.error("Query error: %s", exc)
        return []


@tool
def search_entities(query: str, limit: int = 15) -> str:
    """Find entities whose name contains the given text. Use this to locate the
    exact entity name before calling other tools."""
    rows = execute_query(
        """
        MATCH (e:Entity)
        WHERE lower(e.name) CONTAINS lower($q)
        RETURN e.name AS name, e.type AS type, e.description AS description
        LIMIT $limit
        """,
        {"q": query, "limit": limit},
    )
    if not rows:
        return f"No entities found matching '{query}'."
    out = [f"Entities matching '{query}':"]
    for r in rows:
        desc = f" — {r['description']}" if r.get("description") else ""
        out.append(f"- {r['name']} ({r['type']}){desc}")
    return "\n".join(out)


@tool
def get_entity_relationships(entity_name: str, limit: int = 25) -> str:
    """List the direct relationships of an entity (both incoming and outgoing)."""
    rows = execute_query(
        """
        MATCH (a:Entity {name: $name})-[r:RELATES]-(b:Entity)
        RETURN a.name AS a, r.type AS rel, b.name AS b, b.type AS b_type, r.context AS context
        LIMIT $limit
        """,
        {"name": entity_name, "limit": limit},
    )
    if not rows:
        return f"No relationships found for '{entity_name}'. Try search_entities first."
    out = [f"Relationships for {entity_name}:"]
    for r in rows:
        ctx = f"  ({r['context']})" if r.get("context") else ""
        out.append(f"- {r['a']} -[{r['rel']}]- {r['b']} ({r['b_type']}){ctx}")
    return "\n".join(out)


@tool
def get_neighbors_within_hops(entity_name: str, hops: int = 2, limit: int = 30) -> str:
    """Find entities connected to the given entity within N relationship hops."""
    hops = max(1, min(int(hops), 4))
    rows = execute_query(
        f"""
        MATCH (a:Entity {{name: $name}})-[:RELATES*1..{hops}]-(b:Entity)
        RETURN DISTINCT b.name AS name, b.type AS type
        LIMIT $limit
        """,
        {"name": entity_name, "limit": limit},
    )
    if not rows:
        return f"No entities found within {hops} hops of '{entity_name}'."
    out = [f"Entities within {hops} hop(s) of {entity_name}:"]
    out += [f"- {r['name']} ({r['type']})" for r in rows]
    return "\n".join(out)


@tool
def find_connection(entity_a: str, entity_b: str) -> str:
    """Find how two entities are connected by tracing the shortest path between them."""
    rows = execute_query(
        """
        MATCH p = (a:Entity {name: $a})-[:RELATES* SHORTEST 1..5]-(b:Entity {name: $b})
        RETURN nodes(p) AS nodes, rels(p) AS rels
        LIMIT 1
        """,
        {"a": entity_a, "b": entity_b},
    )
    if not rows:
        return f"No path found between '{entity_a}' and '{entity_b}' (within 5 hops)."
    nodes = [n["name"] for n in rows[0]["nodes"]]
    rels = [r["type"] for r in rows[0]["rels"]]
    path = nodes[0]
    for rel, node in zip(rels, nodes[1:]):
        path += f" -[{rel}]- {node}"
    return f"Path from {entity_a} to {entity_b}:\n{path}"


@tool
def most_connected_entities(limit: int = 10) -> str:
    """Rank the most connected (highest-degree) entities in the graph."""
    rows = execute_query(
        """
        MATCH (e:Entity)-[r:RELATES]-()
        RETURN e.name AS name, e.type AS type, count(r) AS degree
        ORDER BY degree DESC
        LIMIT $limit
        """,
        {"limit": limit},
    )
    if not rows:
        return "No entity relationships found in the graph."
    out = ["Most connected entities:"]
    out += [f"- {r['name']} ({r['type']}): {r['degree']} connections" for r in rows]
    return "\n".join(out)


@tool
def documents_mentioning(entity_name: str, limit: int = 20) -> str:
    """List the source documents that mention a given entity."""
    rows = execute_query(
        """
        MATCH (d:Document)-[:MENTIONS]->(e:Entity {name: $name})
        RETURN d.title AS title, d.source AS source
        LIMIT $limit
        """,
        {"name": entity_name, "limit": limit},
    )
    if not rows:
        return f"No documents mention '{entity_name}'."
    out = [f"Documents mentioning {entity_name}:"]
    out += [f"- {r['title']} ({r['source']})" for r in rows]
    return "\n".join(out)


@tool
def entities_in_document(title: str, limit: int = 40) -> str:
    """List the entities mentioned in a document (matched by title substring)."""
    rows = execute_query(
        """
        MATCH (d:Document)-[:MENTIONS]->(e:Entity)
        WHERE lower(d.title) CONTAINS lower($title)
        RETURN e.name AS name, e.type AS type
        LIMIT $limit
        """,
        {"title": title, "limit": limit},
    )
    if not rows:
        return f"No entities found for a document matching '{title}'."
    out = [f"Entities in documents matching '{title}':"]
    out += [f"- {r['name']} ({r['type']})" for r in rows]
    return "\n".join(out)


def get_retriever_tools(db_path: str) -> List:
    """Return all graph retrieval tools, bound to ``db_path``."""
    initialize_db_connection(db_path)
    return [
        search_entities,
        get_entity_relationships,
        get_neighbors_within_hops,
        find_connection,
        most_connected_entities,
        documents_mentioning,
        entities_in_document,
    ]
