# WORKING.md — Plan for Dataset Generator CLI (LoRA training data)

## Context
We already have (or are building) a LoRA exemplar:
- Ask baseline model (CodeLlama)
- Fine-tune with LoRA/QLoRA using data_train.jsonl
- Ask again
- Judge baseline vs tuned

This new work produces a separate CLI that generates data_train.jsonl and data_eval.jsonl from documents, reducing manual curation.

## Goal
Create a CLI that turns a set of docs into a curated Q&A dataset aligned to a theme/hypothesis (e.g., “HPC user enablement best practices”).

Example hypothesis (Slurm):
Teach reliable Slurm user enablement and day-1 operational best practices: interactive sessions, node discovery, resource requests, and common pitfalls, using safe placeholders for site-specific flags.

## Key idea: agentic pipeline with guardrails
We need more than “LLM writes Q&A.” We need:
1) Curriculum (topic map + rules) driven by hypothesis
2) Grounded generation (answers must be supported by retrieved chunks)
3) Quality gates (judge + deterministic validators)
4) Dedupe + balance (avoid endless paraphrases of the same command)

## Data formats

Corpus record (corpus.jsonl):
- doc_id
- title
- source_path
- license (optional)
- text (normalized)

Chunk record (chunks.jsonl):
- doc_id
- chunk_id (doc_id + sequence number)
- section (best-effort)
- text
- keywords (best-effort)

Candidate Q&A record (candidates.jsonl):
- instruction
- output
- topic (e.g., slurm.interactive)
- difficulty (L1/L2/L3)
- citations (list of chunk_id)
- notes (optional)

Final LoRA dataset:
- train.jsonl: instruction + output
- eval.jsonl: instruction + output (held-out)
Recommendation: keep metadata/provenance in a separate manifest file (manifest.jsonl).

## Prompting strategy (templates)

Curriculum prompt (hypothesis → topic map + rules):
- topics + coverage quotas
- must-include tokens/phrases
- must-avoid patterns

Q generation prompt (chunk → questions):
- generate 2–5 questions per chunk
- align to hypothesis topics
- mix styles: how-to, troubleshooting, explain-why
- tag with topic + difficulty

Answer generation prompt (question + chunk(s) → answer):
- strict grounding: only use the provided chunks
- provide 1–2 examples when relevant
- include placeholders for site specifics
- keep concise

Judge prompt (question + answer + chunks → decision):
- return JSON: groundedness, correctness, clarity, keep/reject
- if fixable, provide a suggested rewrite

## Deterministic validators (non-LLM)
Add cheap checks, especially for Slurm:
- If topic is slurm.interactive, require “srun --pty”
- If question mentions GPU, require “--gres=gpu” or “--gpus”
- Forbid hard-coded partitions/accounts; enforce placeholders
- Max length and formatting checks

These validators complement the judge and reduce drift.

## Dedupe + balance
Dedup, in order:
1) Exact hash dedupe on normalized (instruction, output)
2) Near-duplicate similarity (token-based)
3) Optional embedding similarity if enabled

Balance:
- Ensure each topic has at least N items
- Ensure a mix of L1/L2/L3 (mostly L1/L2)

## CLI design

datasetgen init:
- creates datasetgen.yaml (hypothesis, topics, rules, models, seed)

datasetgen run --input docs/ --hypothesis datasetgen.yaml --outdir out/:
- out/corpus.jsonl
- out/chunks.jsonl
- out/index/
- out/candidates.jsonl
- out/train.jsonl
- out/eval.jsonl
- out/manifest.jsonl (provenance + metrics)

Minimum viable run (even if LLM calls are stubbed):
- ingest docs
- chunk
- build index
- output scaffolding files

## Integration with the LoRA exemplar
Workflow:
1) datasetgen run ...
2) copy out/train.jsonl → LoRA demo data_train.jsonl
3) run LoRA demo (ask → tune → ask → judge)
4) report improvements on held-out eval prompts

## First milestone (Slurm enablement)
Input docs:
- Slurm manpages (srun, salloc, sbatch, sinfo, squeue) converted to text
- site user guide markdown (optional)

Output:
- train: 30–80 Q&As
- eval: 10–20 Q&As
All answers:
- concise
- include 1–2 command examples
- use placeholders for site specifics

## Notes / gotchas
- Licensing: only ingest sources you are allowed to use for training.
- Keep provenance: citations per item to support audits and debugging.
- Avoid overfitting by repetition: don’t generate hundreds of paraphrases.