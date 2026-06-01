# Graph Research

This repository contains research and trial scripts for everything related to graph-based analytics. This includes graph-based optimization, graphRAG, graph databases, and graph neural networks.

## Table of Contents

- [Graph Based Optimization](#graph-based-optimization)
- [Kuzu](#kuzu)
- [GraphRAG](#graphrag)
- [Hierarchy Mapping](#hierarchy-mapping)

## Graph Based Optimization 

A small proof of concept for graph-based optimization workflows. See `graph_optim/`.

## Kuzu
  
Testing scripts for running Kuzu on Databricks and testing authentication methods. See `kuzu/`.

## GraphRAG

A minimal, domain-agnostic GraphRAG pipeline (`graphrag/`): ingest local documents,
extract a knowledge graph with an LLM, store it in Kuzu, and query it with a
LangGraph agent. Ingest → extract → build → query, all from a single CLI.
See [`graphrag/README.md`](graphrag/README.md).

## Hierarchy Mapping

A Databricks demo (`hierarchy_mapping/`) for hierarchy navigation and agent-based
mapping using GraphFrames (batch analytics), recursive CTEs (interactive traversal),
Vector Search (semantic matching), and an LLM agent — demonstrated on six years of
Google Product Taxonomy evolution. See [`hierarchy_mapping/README.md`](hierarchy_mapping/README.md).