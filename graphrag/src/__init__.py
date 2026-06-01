"""GraphRAG — a minimal local-document knowledge graph + retrieval agent."""

__version__ = "0.1.0"

from .models import Document, Entity, Relationship

__all__ = [
    "GraphRAGAgent",
    "create_graphrag_agent",
    "Document",
    "Entity",
    "Relationship",
]


def __getattr__(name):
    # Lazy import so that lightweight modules (ingest, models) don't pull in
    # databricks-langchain / langgraph until the agent is actually used.
    if name in {"GraphRAGAgent", "create_graphrag_agent"}:
        from . import agent

        return getattr(agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
