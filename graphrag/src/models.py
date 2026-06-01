"""Data models for the GraphRAG pipeline."""

from dataclasses import dataclass


@dataclass
class Document:
    """A source document ingested into the knowledge graph."""

    doc_id: str
    title: str
    source: str
    content: str


@dataclass
class Entity:
    """An entity extracted from a document."""

    name: str
    type: str
    description: str = ""


@dataclass
class Relationship:
    """A directed relationship between two entities."""

    source: str
    target: str
    type: str
    context: str = ""
