# datasetgen

Offline-first CLI to generate a curated LoRA dataset (JSONL) from local docs.

## Quick start

```bash
python -m datasetgen.cli init --out datasetgen.yaml
```

Environment
```
OPENAI_API_KEY=sk-...
```

.env
```
OPENAI_API_KEY=sk-...
```
`datasetgen` will automatically load `.env` from the repo root if `python-dotenv` is installed.

## Usage

```
datasetgen init [--out datasetgen.yaml]

datasetgen ingest --input <path>... --out corpus.jsonl [--license <text>]

datasetgen build-index --corpus corpus.jsonl --out index/ [--max-chars 800] [--overlap 120]

datasetgen generate --index index/ --hypothesis datasetgen.yaml --out candidates.jsonl
datasetgen generate --index index/ --hypothesis datasetgen.yaml --out candidates.jsonl --limit 20

datasetgen curate --in candidates.jsonl --out train.jsonl --eval-out eval.jsonl [--manifest manifest.jsonl]
datasetgen curate --in candidates.jsonl --out train.jsonl --eval-out eval.jsonl --judge human --human-review-out review.jsonl
datasetgen curate --in candidates.jsonl --out train.jsonl --eval-out eval.jsonl --human-review-in reviewed.jsonl
datasetgen pretty-print --in train.jsonl --out train.pretty.json
datasetgen pretty-print --in train.jsonl --mode text --limit 5

datasetgen run --input <path>... --hypothesis datasetgen.yaml --outdir out/
```

Options (by command)
- `init --out`: Path to write the config stub (default: `datasetgen.yaml`).
- `ingest --input`: One or more files or directories to ingest (`.md`, `.txt`, `.rst`, `.html`, `.pdf`).
- `ingest --out`: Output corpus JSONL path.
- `ingest --license`: License string recorded in corpus metadata.
- `build-index --corpus`: Input corpus JSONL from `ingest`.
- `build-index --out`: Output index directory.
- `build-index --max-chars`: Max chunk size in characters.
- `build-index --overlap`: Chunk overlap in characters.
- `generate --index`: Index directory from `build-index`.
- `generate --hypothesis`: Config file (YAML) with hypothesis, topics, rules, quotas, model, seed.
- `generate --out`: Output candidates JSONL path.
- `generate --limit`: Max number of generated items (overrides config).
- `generate (config) question_index`: Optional index directory for question generation; defaults to `--index`.
- `generate (config) answer_index`: Optional index directory for answer retrieval; defaults to `--index`.
- `generate (config) question_limit`: Optional max number of generated items (0 = no limit).
- `curate --in`: Input candidates JSONL path.
- `curate --out`: Output train JSONL path.
- `curate --eval-out`: Output held-out eval JSONL path.
- `curate --manifest`: Output provenance manifest path.
- `curate --judge`: Judge mode: `stub` (default), `off`, or `human`.
- `curate --human-review-out`: Write review queue JSONL for human-in-the-loop.
- `curate --human-review-in`: Read reviewed JSONL with `decision`/`revised_output`.
- `curate --index`: Optional index directory used to validate grounding against cited chunks.
- `pretty-print --in`: Input JSONL file to pretty print.
- `pretty-print --out`: Output JSON file (if omitted, prints to stdout).
- `pretty-print --mode`: `json` (default) or `text` for human-friendly output blocks.
- `pretty-print --limit`: Max number of items to print (0 = all).
- `run --input`: One or more files or directories to ingest.
- `run --hypothesis`: Config file (YAML) to use for the full pipeline.
- `run --outdir`: Output directory for all artifacts.

```bash
python -m datasetgen.cli ingest --input docs/ --out out/corpus.jsonl
python -m datasetgen.cli build-index --corpus out/corpus.jsonl --out out/index
python -m datasetgen.cli generate --index out/index --hypothesis datasetgen.yaml --out out/candidates.jsonl
python -m datasetgen.cli curate --in out/candidates.jsonl --out out/train.jsonl --eval-out out/eval.jsonl --manifest out/manifest.jsonl
```

Or run the full pipeline:

```bash
python -m datasetgen.cli run --input docs/ --hypothesis datasetgen.yaml --outdir out/
```

## Output
- `out/corpus.jsonl`: normalized documents
- `out/index/`: chunks + BM25 index
- `out/candidates.jsonl`: generated Q&A with citations
- `out/train.jsonl`: LoRA train dataset
- `out/eval.jsonl`: held-out eval dataset
- `out/manifest.jsonl`: provenance metadata

## Notes
- Default LLM calls are stubbed for determinism (no network).
- Replace the stub with a real LLM client to improve quality.
- OpenAI use: set `model.generator: "openai"` and provide `openai_model` and `openai_api_key` (or `OPENAI_API_KEY` env var).
- HTML ingestion uses BeautifulSoup when available for cleaner text extraction.
- PDF ingestion uses `pypdf` for text extraction.
- For Slurm commands, use placeholders: `-p <partition> -A <account>`.

## Example `datasetgen.yaml`

```yaml
hypothesis: "Teach reliable Slurm user enablement and day-1 operational best practices."
topics:
  slurm.interactive:
    commands: ["srun", "salloc"]
    quota: 12
  slurm.discovery:
    commands: ["sinfo", "squeue"]
    quota: 10
  slurm.batch:
    commands: ["sbatch"]
    quota: 8
rules:
  must_include:
    - "use placeholders for partition/account"
  must_avoid:
    - "hard-coded -p/-A values"
quotas:
  L1: 18
  L2: 10
  L3: 4
model:
  generator: "stub"
  judge: "stub"
  openai_model: ""
  openai_api_key: ""
question_index: ""
answer_index: ""
question_limit: 0
seed: 0
```

Config fields
- `hypothesis`: High-level instruction guiding question/answer style and scope.
- `topics`: Map of topic → settings. Each topic supports:
  - `commands`: List of command keywords used for retrieval.
  - `quota`: Target number of items to generate for the topic.
- `rules`: Guardrails for generation.
  - `must_include`: Phrases or requirements to include.
  - `must_avoid`: Patterns or requirements to avoid.
- `quotas`: Target distribution across difficulty levels (`L1`/`L2`/`L3`).
- `model`: LLM settings.
  - `generator`: `stub` or `openai`.
  - `judge`: `stub`, `off`, or `human`.
  - `openai_model`: OpenAI model name.
  - `openai_api_key`: API key (optional if `OPENAI_API_KEY` is set).
- `question_index`: Index directory to use for question generation (optional).
- `answer_index`: Index directory to use for answer retrieval (optional).
- `question_limit`: Cap on generated items (0 = no limit).
- `seed`: Reproducibility seed.

## Use case: Slurm docs → curated dataset (end to end)

1) Download docs

```bash
mkdir -p docs/questions docs/answers out
curl -L https://slurm.schedmd.com/quickstart.html -o docs/questions/quickstart.html
curl -L https://slurm.schedmd.com/quickstart_admin.html -o docs/questions/quickstart_admin.html
curl -L https://slurm.schedmd.com/srun.html -o docs/answers/srun.html
curl -L https://slurm.schedmd.com/salloc.html -o docs/answers/salloc.html
curl -L https://slurm.schedmd.com/sbatch.html -o docs/answers/sbatch.html
curl -L https://slurm.schedmd.com/sinfo.html -o docs/answers/sinfo.html
curl -L https://slurm.schedmd.com/squeue.html -o docs/answers/squeue.html
```

2) Build indexes

```bash
python -m datasetgen.cli ingest --input docs/questions --out out/q_corpus.jsonl
python -m datasetgen.cli build-index --corpus out/q_corpus.jsonl --out out/q_index
python -m datasetgen.cli ingest --input docs/answers --out out/a_corpus.jsonl
python -m datasetgen.cli build-index --corpus out/a_corpus.jsonl --out out/a_index
```

3) Configure (`datasetgen.yaml`)

```yaml
model:
  generator: "openai"
  openai_model: "gpt-4.1-mini"
question_index: "out/q_index"
answer_index: "out/a_index"
question_limit: 25
```

4) Add `.env`

```
OPENAI_API_KEY=sk-...
```

5) Generate + curate

```bash
python -m datasetgen.cli generate --index out/q_index --hypothesis datasetgen.yaml --out out/candidates.jsonl --limit 25
python -m datasetgen.cli curate --in out/candidates.jsonl --out out/train.jsonl --eval-out out/eval.jsonl --manifest out/manifest.jsonl --index out/a_index
python -m datasetgen.cli pretty-print --in out/train.jsonl --out out/train.pretty.json
```
