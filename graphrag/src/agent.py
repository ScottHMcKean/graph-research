"""LangGraph agent that answers questions over the Kuzu knowledge graph.

A simple tool-calling loop: the LLM decides which graph tools to call, the
tools run against Kuzu, and the loop continues until the LLM produces a final
answer. Domain-agnostic — no dataset-specific query classification.
"""

import logging

from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.query_tools import get_retriever_tools, initialize_db_connection
from src.state import GraphRAGState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions using a knowledge graph of entities and the documents they came from.

You have tools to search entities, inspect their relationships, traverse neighbors,
find paths between entities, rank the most-connected entities, and look up which
documents mention an entity. Plan which tools to call, gather evidence from the
graph, then answer concisely and cite the entities/documents you used.

If an entity name is uncertain, call search_entities first to resolve it."""


class GraphRAGAgent:
    """Tool-calling agent over a Kuzu knowledge graph."""

    def __init__(
        self,
        model: str = "databricks-claude-sonnet-4",
        db_path: str = "build/graph.kuzu",
        temperature: float = 0.1,
    ):
        self.model_name = model
        self.db_path = db_path

        initialize_db_connection(db_path)
        self.tools = get_retriever_tools(db_path)
        self.llm = ChatDatabricks(endpoint=model, temperature=temperature, max_tokens=1500)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        self.checkpointer = MemorySaver()
        self.app = self._build_graph().compile(checkpointer=self.checkpointer)

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphRAGState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", self._should_continue, {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")
        return workflow

    def _agent_node(self, state: GraphRAGState) -> GraphRAGState:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = self.llm_with_tools.invoke(messages)
        return {"messages": messages + [response]}

    def _should_continue(self, state: GraphRAGState) -> str:
        last = state["messages"][-1] if state["messages"] else None
        if last and getattr(last, "tool_calls", None):
            return "continue"
        return "end"

    def query(self, user_query: str, session_id: str = "default") -> str:
        """Answer a single question and return the final text response."""
        try:
            config = {"configurable": {"thread_id": session_id}}
            result = self.app.invoke(
                {"messages": [HumanMessage(content=user_query)]}, config=config
            )
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    return msg.content
            return "I couldn't produce an answer. Please rephrase your question."
        except Exception as exc:  # noqa: BLE001
            logger.error("Error processing query: %s", exc)
            return f"Error while processing your query: {exc}"

    def stream_query(self, user_query: str, session_id: str = "default"):
        """Stream intermediate output (agent tokens and tool results)."""
        try:
            config = {"configurable": {"thread_id": session_id}}
            for chunk in self.app.stream(
                {"messages": [HumanMessage(content=user_query)]}, config=config
            ):
                if "agent" in chunk:
                    msg = chunk["agent"]["messages"][-1]
                    if isinstance(msg, AIMessage) and msg.content:
                        yield msg.content
                elif "tools" in chunk:
                    for msg in chunk["tools"]["messages"]:
                        if isinstance(msg, ToolMessage):
                            yield f"\n[{msg.name}]\n{msg.content}\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("Error streaming query: %s", exc)
            yield f"Error: {exc}"


def create_graphrag_agent(
    model: str = "databricks-claude-sonnet-4",
    db_path: str = "build/graph.kuzu",
    temperature: float = 0.1,
) -> GraphRAGAgent:
    """Factory for a configured GraphRAG agent."""
    return GraphRAGAgent(model=model, db_path=db_path, temperature=temperature)
