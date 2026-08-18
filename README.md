# 🔎 Large-Scale Search Engine + Neural Ranking System

A scalable search engine combining **BM25 lexical retrieval** with **Transformer-based neural reranking** for improved semantic search.

## 🚀 Features

* 🔍 BM25-based candidate retrieval using **Elasticsearch**
* 🧠 Semantic ranking using **Sentence Transformers / BERT**
* 🔀 Hybrid BM25 + neural ranking
* 📊 Evaluation using **Recall@K, MRR, and NDCG**
* ⚡ FastAPI search API
* 💻 React + Tailwind search interface
* 🐳 Dockerized deployment

## 🏗️ Architecture

```text
User Query
    ↓
Elasticsearch / BM25
    ↓
Top-N Candidates
    ↓
Neural Reranker
    ↓
Hybrid Ranking
    ↓
Top-K Results
```

## 🛠️ Tech Stack

**Python · Elasticsearch · BM25 · PyTorch · Transformers · Sentence Transformers · FastAPI · React · Tailwind CSS · Docker**

## 📂 Structure

```text
├── ingestion/      # Data processing & indexing
├── retrieval/      # BM25 retrieval
├── ranking/        # Neural reranking
├── evaluation/     # Search metrics
├── api/            # FastAPI backend
├── frontend/       # React UI
├── data/           # Dataset
└── docker-compose.yml
```

## 🎯 Goal

Build and evaluate a **large-scale two-stage retrieval system** where BM25 provides fast candidate generation and a neural model improves the final ranking through semantic understanding.

**Status:** 🚧 In Development
