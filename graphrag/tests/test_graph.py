import pytest

pytest.importorskip("kuzu")

from src import query_tools
from src.graph_build import build_graph
from src.models import Document, Entity, Relationship


@pytest.fixture
def graph(tmp_path):
    docs = [Document("d1", "Doc One", "d1.md", "text")]
    entities = [
        Entity("Northwind", "ORG", "a robotics company"),
        Entity("Kestrel", "PRODUCT", "a drone"),
        Entity("Meridian", "ORG", "a pipeline operator"),
    ]
    rels = [
        Relationship("Northwind", "Kestrel", "PRODUCES"),
        Relationship("Meridian", "Kestrel", "USES"),
    ]
    mentions = [("d1", "Northwind"), ("d1", "Kestrel")]
    db_path = tmp_path / "graph.kuzu"
    build_graph(db_path, docs, entities, rels, mentions)
    # Reset the module-level singleton so each test opens its own graph.
    query_tools._connection = None
    query_tools.initialize_db_connection(str(db_path))
    yield
    query_tools._connection = None


def test_search_entities(graph):
    out = query_tools.search_entities.invoke({"query": "kestrel"})
    assert "Kestrel" in out


def test_entity_relationships(graph):
    out = query_tools.get_entity_relationships.invoke({"entity_name": "Kestrel"})
    assert "Northwind" in out and "Meridian" in out


def test_find_connection(graph):
    out = query_tools.find_connection.invoke(
        {"entity_a": "Northwind", "entity_b": "Meridian"}
    )
    assert "Kestrel" in out


def test_most_connected(graph):
    out = query_tools.most_connected_entities.invoke({"limit": 3})
    assert "Kestrel" in out  # highest degree node


def test_documents_mentioning(graph):
    out = query_tools.documents_mentioning.invoke({"entity_name": "Northwind"})
    assert "Doc One" in out
