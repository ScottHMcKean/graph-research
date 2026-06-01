#!/usr/bin/env python3
"""GraphRAG pipeline CLI.

Stages:
  ingest   data/docs/*  ->  build/documents.parquet
  extract  documents    ->  build/{entities,relationships,mentions}.parquet   (LLM)
  build    parquet       ->  build/graph.kuzu                                   (Kuzu)
  query    graph         ->  interactive / one-shot Q&A                         (LLM agent)
  all      ingest + extract + build

Examples:
  uv run python main.py all
  uv run python main.py ingest --docs data/docs
  uv run python main.py extract --model databricks-claude-sonnet-4
  uv run python main.py query "How are the entities connected?"
  uv run python main.py query            # interactive REPL
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow `from src...` imports when run as a script.
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BUILD = Path("build")
DOCS_PARQUET = BUILD / "documents.parquet"
ENTITIES_PARQUET = BUILD / "entities.parquet"
RELS_PARQUET = BUILD / "relationships.parquet"
MENTIONS_PARQUET = BUILD / "mentions.parquet"
GRAPH_PATH = BUILD / "graph.kuzu"


def cmd_ingest(args) -> int:
    from src.ingest import ingest_documents, save_documents

    docs = ingest_documents(args.docs)
    if not docs:
        logger.error("No documents ingested from %s", args.docs)
        return 1
    save_documents(docs, DOCS_PARQUET)
    return 0


def cmd_extract(args) -> int:
    from src.extraction import extract_from_documents, save_extraction
    from src.ingest import load_documents

    if not DOCS_PARQUET.exists():
        logger.error("Run `ingest` first (missing %s).", DOCS_PARQUET)
        return 1
    docs = load_documents(DOCS_PARQUET)
    entities, relationships, mentions = extract_from_documents(docs, model=args.model)
    if not entities:
        logger.error("No entities extracted; check the model endpoint and documents.")
        return 1
    save_extraction(
        entities, relationships, mentions,
        str(ENTITIES_PARQUET), str(RELS_PARQUET), str(MENTIONS_PARQUET),
    )
    return 0


def cmd_build(args) -> int:
    from src.extraction import load_extraction
    from src.graph_build import build_graph
    from src.ingest import load_documents

    for required in (DOCS_PARQUET, ENTITIES_PARQUET, RELS_PARQUET, MENTIONS_PARQUET):
        if not required.exists():
            logger.error("Missing %s. Run ingest + extract first.", required)
            return 1
    docs = load_documents(DOCS_PARQUET)
    entities, relationships, mentions = load_extraction(
        str(ENTITIES_PARQUET), str(RELS_PARQUET), str(MENTIONS_PARQUET)
    )
    build_graph(GRAPH_PATH, docs, entities, relationships, mentions)
    return 0


def cmd_query(args) -> int:
    from src.agent import GraphRAGAgent

    if not GRAPH_PATH.exists():
        logger.error("No graph found at %s. Run `all` first.", GRAPH_PATH)
        return 1
    agent = GraphRAGAgent(model=args.model, db_path=str(GRAPH_PATH))

    if args.question:
        print(agent.query(" ".join(args.question)))
        return 0

    print("GraphRAG interactive query. Type 'quit' to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"quit", "exit", "q", ""}:
            break
        print(agent.query(q))
    return 0


def cmd_all(args) -> int:
    return cmd_ingest(args) or cmd_extract(args) or cmd_build(args)


def main() -> int:
    parser = argparse.ArgumentParser(description="GraphRAG pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest local documents")
    p_ingest.add_argument("--docs", default="data/docs", help="Documents directory")

    p_extract = sub.add_parser("extract", help="LLM entity/relationship extraction")
    p_extract.add_argument("--model", default="databricks-claude-sonnet-4")

    sub.add_parser("build", help="Build the Kuzu graph")

    p_query = sub.add_parser("query", help="Query the graph (interactive if no question)")
    p_query.add_argument("question", nargs="*", help="Question to ask (optional)")
    p_query.add_argument("--model", default="databricks-claude-sonnet-4")

    p_all = sub.add_parser("all", help="ingest + extract + build")
    p_all.add_argument("--docs", default="data/docs")
    p_all.add_argument("--model", default="databricks-claude-sonnet-4")

    args = parser.parse_args()
    BUILD.mkdir(exist_ok=True)

    return {
        "ingest": cmd_ingest,
        "extract": cmd_extract,
        "build": cmd_build,
        "query": cmd_query,
        "all": cmd_all,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
