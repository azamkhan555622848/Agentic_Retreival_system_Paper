 Implementation Plan: Agentic RAG with Hybrid Retrieval Experimentation

 Context

 This project implements the full experimentation pipeline described in the research PDF "Agentic RAG with Hybrid Retrieval." The core novelty is an agentic RAG system where an LLM agent       
 dynamically chooses between grep-based (regex/lexical) retrieval and semantic (dense vector) retrieval as tools — something no existing academic paper has systematically evaluated. The        
 project is a greenfield build (only the PDF exists currently) targeting a Q2/Q1 journal publication from NYCU.

 Hardware: 3x NVIDIA RTX A6000 (48GB each, 144GB total)
 Agent Framework: LangGraph
 LLMs: Gemma 2 27B, Qwen2.5-7B/14B/32B, Llama 3.1 8B (baseline)
 Scope: Full pipeline — 6 retrieval configs x 5 models x 5 benchmarks, evaluated with RAGAS + ARES

 ---
 Project Structure

 D:/Agentic_RAG_Research_Paper/
 ├── pyproject.toml
 ├── .env                          # HF_TOKEN, API keys (gitignored)
 ├── .gitignore
 ├── configs/
 │   ├── base.yaml                 # Shared defaults (paths, seeds, GPU IDs)
 │   ├── models/                   # Per-model configs (gemma2_27b.yaml, qwen25_7b.yaml, etc.)
 │   ├── retrievers/               # Per-retriever configs (grep_only.yaml, grep_semantic.yaml, etc.)
 │   ├── benchmarks/               # Per-benchmark configs (hotpotqa.yaml, musique.yaml, etc.)
 │   ├── evaluation/               # RAGAS + ARES configs
 │   └── experiment_matrix.yaml    # Master matrix: all model x retriever x benchmark combos
 ├── data/
 │   ├── raw/                      # Downloaded datasets from HuggingFace
 │   ├── corpus/                   # Wikipedia corpus JSONL + sharded files for grep
 │   ├── indexes/                  # FAISS (BGE-M3) and BM25 indexes
 │   ├── processed/                # FlashRAG-format processed datasets
 │   └── annotations/              # Human annotations for ARES (150-300 per dataset)
 ├── src/
 │   ├── config/                   # Config loading (loader.py) + Pydantic validation (schema.py)
 │   ├── retrievers/               # All retriever implementations
 │   │   ├── base.py               # BaseRetriever ABC + RetrievedDocument/RetrievalResult dataclasses
 │   │   ├── grep_retriever.py     # GrepRetriever: ripgrep over sharded JSONL corpus
 │   │   ├── query_to_regex.py     # LLM-powered natural language → regex converter (KEY NOVELTY)
 │   │   ├── dense_retriever.py    # DenseRetriever: FAISS + BGE-M3
 │   │   ├── bm25_retriever.py     # BM25Retriever: bm25s library
 │   │   ├── hybrid_retriever.py   # HybridRetriever: multi-source + RRF fusion
 │   │   └── fusion.py             # Reciprocal Rank Fusion implementation
 │   ├── agent/                    # LangGraph agentic RAG
 │   │   ├── state.py              # AgentState TypedDict
 │   │   ├── tools.py              # Tool factory: wraps retrievers as LangGraph tools
 │   │   ├── nodes.py              # Node functions: agent reasoning, grading, rewriting, generation
 │   │   ├── graph.py              # StateGraph construction and compilation
 │   │   └── prompts.py            # All prompt templates
 │   ├── evaluation/               # All evaluation logic
 │   │   ├── retrieval_metrics.py  # Recall@K, nDCG@10, MRR
 │   │   ├── generation_metrics.py # EM, F1, BERTScore, ROUGE-L
 │   │   ├── ragas_evaluator.py    # RAGAS wrapper (faithfulness, relevancy, precision, recall)
 │   │   ├── ares_evaluator.py     # ARES PPI wrapper with confidence intervals
 │   │   ├── efficiency_metrics.py # Latency, token count, retrieval calls
 │   │   └── statistical_tests.py  # Paired bootstrap, Wilcoxon, McNemar
 │   ├── data/                     # download.py, preprocess.py, corpus_builder.py, index_builder.py
 │   ├── models/                   # vllm_server.py (launcher), model_registry.py (HF IDs + GPU map)
 │   └── utils/                    # logging, GPU monitoring, timing decorators, IO helpers
 ├── scripts/
 │   ├── download_data.py          # Download all datasets + Wikipedia corpus
 │   ├── build_indexes.py          # Build FAISS + BM25 indexes
 │   ├── run_experiment.py         # Run single (model, retriever, benchmark) experiment
 │   ├── run_matrix.py             # Run full experiment matrix
 │   ├── run_evaluation.py         # Compute all metrics on experiment results
 │   ├── run_ares.py               # ARES-specific evaluation pipeline
 │   ├── generate_tables.py        # LaTeX tables from results
 │   ├── generate_plots.py         # Paper figures (PDF/SVG)
 │   └── ablation_analysis.py      # Ablation study analysis
 ├── notebooks/                    # Jupyter notebooks for exploration & debugging
 ├── results/
 │   ├── raw/{model}_{retriever}_{benchmark}_{timestamp}/  # Per-run outputs
 │   ├── aggregated/               # main_results.csv, retrieval_ablation.csv
 │   ├── tables/                   # LaTeX .tex files
 │   └── figures/                  # Publication-ready plots
 └── tests/                        # Unit + integration tests for retrievers, agent, evaluation


 ---
 Implementation Phases

 Phase 0: Project Scaffolding (Days 1-2)

 - Create full directory structure
 - Write pyproject.toml with all dependencies
 - Write .gitignore, init git repo
 - Write src/config/schema.py — Pydantic models for all configs (ExperimentConfig, ModelConfig, RetrieverConfig, BenchmarkConfig, etc.)
 - Write src/config/loader.py — merges base + model + retriever + benchmark YAMLs
 - Write src/utils/ — logging, GPU monitoring, timing decorators, JSONL I/O

 Key dependencies:

 langgraph>=0.2, langchain>=0.3, vllm>=0.6, torch>=2.4,
 faiss-gpu>=1.7.4, sentence-transformers>=3.0, FlagEmbedding>=1.2,
 bm25s>=0.2, flashrag>=0.2, ragas>=0.2, ares-ai>=0.5,
 bert-score>=0.3.13, rouge-score>=0.1.2, ripgrepy>=2.0,
 pydantic>=2.0, pandas>=2.0, matplotlib>=3.8, seaborn>=0.13, scipy>=1.12


 Phase 1: Data Pipeline (Days 3-7)

 - Write src/data/download.py — download FlashRAG datasets from HuggingFace (RUC-NLPIR/FlashRAG_datasets)
 - Write src/data/preprocess.py — convert to standardized JSONL format
 - Write src/data/corpus_builder.py — build Wikipedia corpus JSONL + shard into ~100 files for parallel grep
 - Write scripts/download_data.py entry point
 - Download & verify all 5 datasets: HotpotQA, MuSiQue, 2WikiMultihopQA, Natural Questions, TriviaQA

 Phase 2: Index Construction (Days 8-12)

 - Write src/data/index_builder.py
 - FAISS index: Encode 21M Wikipedia passages with BGE-M3 → build IndexIVFPQ (product quantization to ~2GB, fits in VRAM). Train on 500K sample, nlist=4096, m=64 subquantizers. ~10 hours on    
 A6000.
 - BM25 index: Tokenize corpus, build bm25s inverted index
 - Corpus sharding: Split wiki_corpus.jsonl into ~100 shards for parallel grep
 - Write scripts/build_indexes.py entry point

 Phase 3: Retriever Implementations (Days 13-20)

 Files to create (in order):

 1. src/retrievers/base.py — BaseRetriever ABC with retrieve(), batch_retrieve(), timed_retrieve(). Dataclasses: RetrievedDocument (doc_id, content, score, rank, source) and RetrievalResult    
 (query, documents, latency_ms, retriever_name).
 2. src/retrievers/dense_retriever.py — DenseRetriever: loads BGE-M3 via FlagEmbedding.BGEM3FlagModel, loads FAISS index to GPU, encodes queries, searches, returns documents with content.      
 3. src/retrievers/bm25_retriever.py — BM25Retriever: loads bm25s index, tokenizes queries, retrieves, maps indices back to document IDs and content.
 4. src/retrievers/query_to_regex.py — QueryToRegexConverter: LLM chain (ChatPromptTemplate → LLM → JsonOutputParser) that converts natural language queries into 1-3 regex patterns. Includes   
 robust fallback (keyword-based patterns) when LLM output is invalid. Validates all patterns compile.
 5. src/retrievers/grep_retriever.py — GrepRetriever: uses ripgrep subprocess (rg --json) to search sharded corpus. For each query: (1) convert to regex via QueryToRegexConverter, (2) search   
 all shards with each pattern, (3) parse ripgrep JSON output, (4) score by multi-pattern match count, (5) return top-k. Includes Python re fallback.
 6. src/retrievers/fusion.py — reciprocal_rank_fusion(): combines multiple ranked lists with RRF (k=60).
 7. src/retrievers/hybrid_retriever.py — HybridRetriever: takes list of sub-retrievers, queries each, fuses via RRF, returns unified ranked list.

 Phase 4: LangGraph Agent (Days 21-30)

 The core agentic component:

 1. src/agent/state.py — AgentState TypedDict with: messages, question, retrieved_documents, retrieval_count, max_retrievals, current_query, tool_calls_log, final_answer.
 2. src/agent/prompts.py — Four prompt templates:
   - AGENT_SYSTEM_PROMPT: instructs agent to break multi-hop questions into sub-queries, choose grep for exact entities / semantic for conceptual queries
   - GRADER_PROMPT: binary relevance grading (relevant/irrelevant)
   - REWRITER_PROMPT: query rewriting for failed retrievals
   - GENERATOR_PROMPT: answer generation from context only
 3. src/agent/tools.py — Tool factory functions: make_grep_tool(), make_dense_tool(), make_bm25_tool(). Each wraps a retriever into a @tool-decorated function with descriptive docstrings that  
 guide tool selection. get_tools_for_config() maps config names to tool lists:
   - grep_only → [grep_search]
   - semantic_only → [semantic_search]
   - bm25_only → [keyword_search]
   - bm25_dense → [keyword_search, semantic_search]
   - grep_semantic → [grep_search, semantic_search] (novel)
   - blended_rag → [keyword_search, semantic_search, grep_search]
 4. src/agent/nodes.py — Node functions:
   - generate_query_or_respond: LLM with bound tools decides to retrieve or answer
   - grade_documents: LLM grades retrieved docs as relevant/irrelevant, routes accordingly
   - rewrite_question: LLM rewrites query for better retrieval
   - generate_answer: LLM generates final answer from accumulated context
 5. src/agent/graph.py — build_agent_graph(llm, tools, max_retrievals=3):
 START → generate_query_or_respond
   ├─(tool call)→ retrieve (ToolNode) → grade_documents
   │                                      ├─(relevant)→ generate_answer → END
   │                                      └─(irrelevant)→ rewrite_question → generate_query_or_respond
   └─(no tool)→ END
 6. src/models/vllm_server.py — Start/stop vLLM server as subprocess. GPU assignments:
   - 7B/8B/14B: single GPU (GPU 0), tensor_parallel=1
   - 27B: single GPU (GPU 0), tensor_parallel=1 (fits in 48GB fp16)
   - 32B: two GPUs (GPU 0+1), tensor_parallel=2 or AWQ quantized on single GPU
 7. src/models/model_registry.py — Maps model names to HuggingFace IDs:
   - gemma2_27b → google/gemma-2-27b-it
   - qwen25_7b → Qwen/Qwen2.5-7B-Instruct
   - qwen25_14b → Qwen/Qwen2.5-14B-Instruct
   - qwen25_32b → Qwen/Qwen2.5-32B-Instruct
   - llama31_8b → meta-llama/Llama-3.1-8B-Instruct

 Phase 5: Experiment Runner (Days 31-38)

 1. scripts/run_experiment.py — Single experiment entry point:
 python scripts/run_experiment.py --model qwen25_14b --retriever grep_semantic --benchmark hotpotqa --max_samples 2000
 1. Flow: load configs → start vLLM → init retrievers → build agent graph → iterate questions → save predictions.jsonl, retrieval_log.jsonl, agent_traces.jsonl, metrics.json
 2. scripts/run_matrix.py — Full matrix runner. Groups by model to minimize vLLM restarts. Checkpoints every 100 questions. Supports resume on failure.
 3. configs/experiment_matrix.yaml — Defines: 5 models x 6 retriever configs x 5 benchmarks = 150 runs

 Subsampling strategy (to keep compute feasible):
 - HotpotQA: 2,000 from dev (stratified by difficulty)
 - MuSiQue: full dev set (2,417)
 - 2WikiMultihopQA: 2,000 from dev
 - NQ: 1,000 / TriviaQA: 1,000
 - Total: ~8,400 questions per config, ~10s per question → ~23 hours per config

 Phase 6: Evaluation Pipeline (Days 39-48)

 1. src/evaluation/retrieval_metrics.py — Recall@5/10/20, nDCG@10, MRR
 2. src/evaluation/generation_metrics.py — EM (normalized), F1 (token-level), BERTScore, ROUGE-L
 3. src/evaluation/ragas_evaluator.py — RAGAS wrapper: faithfulness, answer_relevancy, context_precision, context_recall. Uses local vLLM-served model as judge (avoid OpenAI costs).
 4. src/evaluation/ares_evaluator.py — ARES PPI wrapper: evaluates context relevance, answer faithfulness, answer relevance with confidence intervals. Requires 150-300 human annotations per    
 dataset (600 total across 3 core benchmarks).
 5. src/evaluation/efficiency_metrics.py — Latency per query, tokens consumed, retrieval calls per query
 6. src/evaluation/statistical_tests.py — Paired bootstrap (p<0.05), Wilcoxon signed-rank, McNemar's test

 Phase 7: Full Experiment Execution (Days 49-65)

 - Run all 150 configs (parallelized: 2 configs at a time across GPUs)
 - Run RAGAS + ARES evaluation on all results
 - Aggregate into results/aggregated/main_results.csv

 Phase 8: Analysis & Paper Artifacts (Days 66-75)

 - Tables: Main results (EM/F1 per config per benchmark per model), retrieval quality, RAG triad, ARES with CIs, efficiency, model scaling
 - Figures: Radar charts, bar charts with error bars, heatmaps, model scaling lines, box plots, tool selection Sankey diagram, latency-vs-F1 scatter
 - Statistical significance tests for all pairwise comparisons

 ---
 Known Challenges & Mitigations

 ┌──────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────┐
 │                        Challenge                         │                          Mitigation                          │
 ├──────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ Grep over 21M passages is slow (~25GB corpus)            │ Shard corpus into ~100 files, search in parallel via ripgrep │
 ├──────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ LLM may generate poor regex patterns                     │ Robust keyword fallback + log regex quality as sub-study     │
 ├──────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ FAISS index (21M x 1024d) = 84GB raw                     │ Use IndexIVFPQ (~2GB) or HNSW on CPU                         │
 ├──────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ 150 runs x 8K questions = weeks of compute               │ Subsample to 2K per benchmark, parallelize across GPUs       │
 ├──────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ ARES needs human annotations                             │ Budget 600 annotations (200 per core benchmark)              │
 ├──────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ vLLM tensor parallelism may hang on A6000 without NVLink │ Test early; fallback to AWQ quantization for 32B models      │
 └──────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────┘

 ---
 Verification Plan

 1. Unit tests: Each retriever returns correctly shaped RetrievalResult on 10 sample queries
 2. Integration test: Agent graph completes end-to-end on 10 HotpotQA questions for each retriever config
 3. Pilot run: Run full pipeline on 50 questions per benchmark with 1 model (Qwen2.5-7B) and all 6 retriever configs — verify metrics computation, result serialization, and evaluation pipeline 
 4. Smoke test evaluation: Run RAGAS on pilot results, verify scores are in valid ranges
 5. Index validation: Query FAISS/BM25 with known questions, verify expected documents appear in top-20
 6. Grep validation: Verify ripgrep finds known entity passages, measure per-query latency