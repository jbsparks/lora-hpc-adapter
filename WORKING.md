# WORKING.md — Project handoff / plan

## Goal (exemplar demo)
Create a simple end-to-end example showing:

- Baseline: CodeLlama answers an HPC question.
- Tune: Fine-tune CodeLlama using LoRA/QLoRA on a small HPC mini-dataset.
- After: Ask the same question again and observe improvement.
- Validate: A second LLM (“judge”) scores the answers against a rubric and returns JSON.

This is meant to be a demo artifact (clear narrative + reproducible steps), not an exhaustive benchmark.

---

## What LoRA is (and why we use it here)
**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning method where we keep the **base model frozen** and train a small **adapter** that represents the domain-specific change in behavior.

In this project, LoRA is the “improvement step”:
- **Baseline** = CodeLlama (unchanged foundation model)
- **After tuning** = CodeLlama + a small LoRA adapter (“HPC skill pack”)
- Benefit: fast training, lower GPU memory, and modular adapters (e.g., Slurm vs MPI vs Spack).

If VRAM is limited, use **QLoRA** (LoRA with 4-bit quantized base weights) to make tuning practical on smaller GPUs.

---

## Suggested demo question
“What is the Slurm `srun` command to start an interactive session? Give a minimal example and one requesting 4 CPUs and 16GB for 1 hour.”

### Desired “good” answer traits
- Uses `srun --pty bash`
- Provides a resource-request example: `-t`, `-c`, `--mem`
- Notes site-specific needs: `-p <partition>`, `-A <account>` (as caveats)
- Avoids guessing site-specific partitions/accounts; uses placeholders.

---

## Models

### Base (to tune)
- `codellama/CodeLlama-7b-Instruct-hf`

### Judge (separate model)
Use any local instruction model you can run that is reasonably strong at evaluation, e.g.:
- `meta-llama/Meta-Llama-3.1-8B-Instruct` (if available)
- Another instruct model you already have access to

---

## Data
Create `data_train.jsonl` (JSONL) with records:
- `instruction`: the question
- `output`: the ideal answer

Keep it tightly scoped (Slurm/HPC ops). Include variations:
- interactive shell
- interactive GPU session
- node availability (`sinfo`)
- “scale up vs scale out”
- SSH key auth error explanation
- basic Spack grep multi-pattern usage

Create 5–10 held-out evaluation prompts in `data_eval.jsonl` that are NOT present in training.

---

## Scripts and expected usage

### 1) Baseline inference
```bash
python ask.py "codellama/CodeLlama-7b-Instruct-hf" "<QUESTION>" > baseline.txt
```

### 2) Fine-tune (QLoRA recommended)
```bash
python finetune_lora.py
# outputs adapter dir, e.g. ./codellama-hpc-lora
```

### 3) Inference with adapter
```bash
python ask_lora.py codellama-hpc-lora "<QUESTION>" > tuned.txt
```

### 4) Judge comparison
```bash
python judge.py "<JUDGE_MODEL>" "<QUESTION>" "$(cat baseline.txt)" "$(cat tuned.txt)"
```

## Evaluation rubric (judge)

Score 1–5 on:
	1.	Technical correctness
	2.	Completeness (requested examples present)
	3.	Clarity/conciseness
	4.	Practicality (mentions placeholders, avoids unsafe guesses)

Return JSON only:
```json
{
  "score_a": 0,
  "score_b": 0,
  "winner": "A|B|Tie",
  "rationale": "...",
  "issues_a": ["..."],
  "issues_b": ["..."]
}
```

## Notes for credibility

* Don’t train on the exact wording of every evaluation prompt.
* Show at least one held-out improvement (same topic, different phrasing).
* Keep generation deterministic (do_sample=False) so diffs are clear.