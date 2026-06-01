from typing import Annotated, List, TypedDict

from langgraph.graph.message import add_messages


class GraphRAGState(TypedDict):
    """Conversation state for the GraphRAG agent."""

    messages: Annotated[List, add_messages]
