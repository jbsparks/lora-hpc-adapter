# AGENTS.md — Codex instructions for this repo

## Mission
Implement a minimal, reproducible exemplar that demonstrates a **model-improvement lifecycle** in an HPC context:

1) Ask a **foundation model** (baseline).
2) Fine-tune the model with **LoRA/QLoRA** using a small curated dataset.
3) Ask the **same question** again (post-tune).
4) Use a **second LLM as a judge** to validate/score the baseline vs tuned answers.

## Constraints
- Prefer **CodeLlama Instruct** as the base model.
- Keep the demo **small and fast** (tiny dataset, short runs), but not misleading.
- Avoid network calls in code unless explicitly required.
- Provide clear commands to run each step.
- Keep outputs deterministic where possible (`do_sample=False`).
- Do not assume specific GPU availability; include notes for limited VRAM/CPU-only where helpful.

## Deliverables
- `data_train.jsonl`: small HPC training set (20–50 examples ideal; minimum 8).
- Scripts:
  - `ask.py`: run baseline inference on a prompt/question.
  - `finetune_lora.py`: LoRA/QLoRA supervised fine-tune (PEFT + TRL SFT).
  - `ask_lora.py`: run inference with the LoRA adapter.
  - `judge.py`: compare baseline vs tuned answer using a judge model and a rubric; output JSON.
- `requirements.txt` + brief `README.md` with end-to-end commands.

## Quality bar
- The tuned answer should be **more correct and/or more consistent** than baseline for:
  - the demo question, and
  - at least a few held-out questions (5–10) NOT used in training.

## Style
- Favor concise, operations-ready HPC language.
- For Slurm/Spack/MPI guidance: include 1–2 examples + brief caveats about site-specific flags.
- Do not guess site-specific partitions/accounts; use placeholders (`-p <partition>`, `-A <account>`).

## Suggested default demo topic
Slurm interactive sessions + basic cluster discovery (e.g., `srun --pty`, `sinfo -N -t idle`).

## Safety / correctness checks
- Do not suggest destructive commands.
- When generating commands, be explicit about assumptions and placeholders.