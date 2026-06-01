# GraphRAG

A minimal, domain-agnostic GraphRAG pipeline: ingest a folder of documents,
extract a knowledge graph with an LLM, store it in [Kuzu](https://kuzudb.com/),
and answer questions with a LangGraph agent that queries the graph.

This is a stripped-down rewrite focused purely on the GraphRAG parts. The old
web scraper, RDF/SPARQL layer, and Streamlit app have been removed; ingestion now
reads local files.

## Pipeline

```
data/docs/*.{md,txt,pdf}
        │  ingest
        ▼
build/documents.parquet
        │  extract  (LLM via ChatDatabricks)
        ▼
build/{entities,relationships,mentions}.parquet
        │  build
        ▼
build/graph.kuzu   ──  Document ─[MENTIONS]→ Entity ─[RELATES]→ Entity
        │  query   (LangGraph tool-calling agent)
        ▼
answers
```

## Quick start

```bash
uv sync --extra dev

# Run the whole pipeline on the bundled sample corpus (data/docs/)
uv run python main.py all

# Ask questions
uv run python main.py query "How is Northwind Robotics connected to Meridian Energy?"
uv run python main.py query            # interactive REPL
```

Run stages individually:

```bash
uv run python main.py ingest  --docs data/docs
uv run python main.py extract --model databricks-claude-sonnet-4
uv run python main.py build
```

## Ingesting your own data

Drop `.md`, `.txt`, or `.pdf` files into `data/docs/` (or point `--docs` at any
folder) and re-run `python main.py all`. No web scraping, no network ingestion.

## Requirements

- Python 3.12+ and `uv`
- A Databricks Foundation Model endpoint for extraction and querying
  (`extract` and `query` call `ChatDatabricks`). Set up auth with the Databricks
  CLI / SDK before running those stages.

## Graph tools

The agent has these retrieval tools (`src/query_tools.py`), all generic over the
Document/Entity schema:

| Tool | Purpose |
|------|---------|
| `search_entities` | Resolve an entity by name substring |
| `get_entity_relationships` | Direct neighbors of an entity |
| `get_neighbors_within_hops` | N-hop neighborhood |
| `find_connection` | Shortest path between two entities |
| `most_connected_entities` | Degree-centrality ranking |
| `documents_mentioning` | Which documents mention an entity |
| `entities_in_document` | Entities mentioned in a document |

## Tests

```bash
uv run pytest          # ingest/extraction/graph tests run offline (no LLM)
```

`test_graph.py` is skipped automatically if Kuzu is not installed.
