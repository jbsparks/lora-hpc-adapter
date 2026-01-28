# AGENTS.md — Instructions for Codex (Dataset Generator CLI)

## Mission
Build a new CLI tool that **generates a curated LoRA training dataset (JSONL)** from documentation sources, using an **agentic pipeline**:

1) Ingest / load documents (local files first; optional web later).
2) Chunk + index documents.
3) Generate candidate Q&A pairs grounded in retrieved chunks and aligned to an **overarching hypothesis** (e.g., “HPC user enablement best practices”).
4) Judge/filter/repair candidates (grounding + quality + safety).
5) Deduplicate + balance topic coverage.
6) Export `data_train.jsonl` (and optionally `data_eval.jsonl`) suitable for LoRA fine-tuning.

This tool is a companion to the existing LoRA exemplar (ask → tune → ask → judge), but **kept as a separate module/CLI**.

## Constraints
- Default to **offline/local** operation (no web crawling unless explicitly enabled).
- Use **deterministic** behavior when possible: stable chunking, consistent prompt templates, reproducible random seeds.
- Prioritize dataset **quality and provenance** over quantity.
- Avoid hallucinations by enforcing **grounded generation**:
  - Every answer must be supported by one or more source chunks.
  - Preserve citations to `doc_id/chunk_id`.
- Safety:
  - No destructive commands.
  - For Slurm: do not guess site-specific `partition/account`; use placeholders (`-p <partition>`, `-A <account>`).

## Deliverables
Create a new directory (suggested): `datasetgen/`

1) `datasetgen/cli.py`  
   - Entry point `datasetgen` with subcommands.
2) `datasetgen/pipeline.py`  
   - Orchestration of ingest → chunk → retrieve → generate → judge → dedupe → export.
3) `datasetgen/ingest.py`  
   - Load `.md`, `.txt`, `.rst` (optionally `.pdf` if available via dependency).
4) `datasetgen/chunking.py`  
   - Chunk text + attach metadata (headers, command names).
5) `datasetgen/index.py`  
   - Simple BM25/keyword index first (fast). Embeddings optional behind a flag.
6) `datasetgen/prompts.py`  
   - Prompt templates for: curriculum (hypothesis), qgen, answer generation, judge.
7) `datasetgen/judge.py`  
   - LLM judge + deterministic validators + auto-repair loop.
8) `datasetgen/dedupe.py`  
   - Near-duplicate removal (hash + optional embeddings similarity).
9) `datasetgen/export.py`  
   - Write JSONL (train/eval), plus a manifest with provenance.
10) `README.md`
   - End-to-end examples for generating a Slurm dataset and feeding it into the LoRA demo.

## CLI requirements
Implement at least these commands:

- `datasetgen init`
  - creates a config stub `datasetgen.yaml` with hypothesis, topics, and rules.

- `datasetgen ingest --input <path>... --out corpus.jsonl`
  - loads and normalizes docs into a corpus with metadata.

- `datasetgen build-index --corpus corpus.jsonl --out index/`
  - produces chunk store + retrieval index.

- `datasetgen generate --index index/ --hypothesis datasetgen.yaml --out candidates.jsonl`
  - generates candidate Q&As with citations.

- `datasetgen curate --in candidates.jsonl --out train.jsonl --eval-out eval.jsonl`
  - judge/filter/repair/dedupe/balance; outputs LoRA-ready datasets.

Optional convenience:
- `datasetgen run --input docs/ --hypothesis datasetgen.yaml --outdir out/`
  - runs the full pipeline.

## Config format (datasetgen.yaml)
Must support:
- hypothesis statement
- topic map + target commands
- rules (must include, must avoid)
- quotas per topic and difficulty
- model settings (generator model + judge model)
- reproducibility seed

## Implementation notes
- Keep model calling abstract behind `LLMClient` interface so we can swap:
  - local inference (HF Transformers)
  - OpenAI API (optional, behind flag)
- Start with local-only skeleton that runs end-to-end even if the LLM calls are stubbed.

## Success criteria
- A user can point the tool at a folder of Slurm docs/manpages (text/markdown) and produce:
  - `train.jsonl` (>= 30 high-quality items)
  - `eval.jsonl` (>= 10 held-out items)
  - each item includes provenance metadata
- The dataset improves the LoRA exemplar on at least one held-out question.