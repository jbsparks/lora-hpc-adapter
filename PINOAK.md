# `pinoak` Use Case

## Select a node
```bash
srun --network=single_node_vni  -p rhgriz512 --nodes=1 --exclusive --time=04:00:00 --pty bash
```

## Setup the environment
```bash
cd /lus/cflus02/jsparks/workarea/
mkdir LoRA && cd LoRA
git clone  https://github.com/jbsparks/lora-hpc-adapter
cd lora-hpc-adapter
```

## Run the Examples
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/home/users/${USER}/.local/bin
uvx python@3.13.7 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Should have python set now
```bash
which python
/lus/cflus02/jsparks/workarea/LoRA/lora-hpc-adapter/.venv/bin/python
python --version
Python 3.13.7
```

```bash
mkdir -p /lus/cflus02/jsparks/.cache/huggingface
export HF_HOME=/lus/cflus02/jsparks/.cache/huggingface
# optional: keep Transformers cache consistent
export TRANSFORMERS_CACHE=/lus/cflus02/jsparks/.cache/huggingface/hub
python hpc_lora_cli.py baseline --model-id codellama/CodeLlama-7b-Instruct-hf --out baseline.txt
```
If you want to see the debug/progress simple add `--debug` and/or `--progress` to the CLI.

**note**: the model output may be trucated as we do set the number of returned tokens. the `default` is `512`, but can be controlled by setting `--max-new-tokens 1024`. **OR** update the context to limit the response `-use-context context_default.jsonl`

### Check the output...
_snippet_  `baseline.txt`
```bash
The Slurm srun command to start an interactive session is:

srun --pty --time=0-00:01 --mem=16000 --cpus-per-task=4 --ntasks=1 --nodes=1 --ntasks-per-node=1 --job-name=interactive_session --output=interactive_session.out --error=interactive_session.err --pty bash

This command will start an interactive session with 4 CPUs and 16GB of memory for 1 hour. The output and error files will be named `interactive_session.out` and `interactive_session.err`, respectively.
...
```

## Fine-tuning
```bash
python hpc_lora_cli.py finetune --model-id codellama/CodeLlama-7b-Instruct-hf --output-dir codellama-hpc-lora --debug
```

## Use the LoRA adapter with the same question
```bash
python hpc_lora_cli.py tuned --adapter-path codellama-hpc-lora --out tuned.txt --debug
```

_verify_ the `tuned.txt` model output
```bash
Interactive session:

srun -p <partition> -A <account> -t 00:30:00 -c 1 -A <account> -p <partition> -A <account> -I

4 CPUs and 16GB for 1 hour:

srun -p <partition> -A <account> -t 01:00:00 -c 4 -A <account> -p <partition> -A <account> -I --mem=16G

Note: replace <partition> and <account> with appropriate values for your site.
```

## Judge the responses....

**note**: Use the same model for `judging`...

```bash

python hpc_lora_cli.py judge     --judge-model $HF_HOME/hub/models--codellama--CodeLlama-7b-Instruct-hf/snapshots/22cb240e0292b0b5ab4c17ccd97aa3a2f799cbed     --answer-a "$(cat baseline.txt)" --answer-b "$(cat tuned.txt)" --out judge.json --debug
```

Well ... just goes to show training can cause the opposite affect -- bad answers.

cat `judge.json`
```json
{
  "score_a": 4,
  "score_b": 3,
  "winner": "A",
  "rationale": "Answer A is more complete and provides a more detailed example of how to use the `srun` command to start an interactive session with Slurm. It also includes a request for 4 CPUs and 16GB of memory for 1 hour, which is a more practical example than the one provided in Answer B.",
  "issues_a": [],
  "issues_b": [
    "Request for 4 CPUs and 16GB of memory for 1 hour is not explicitly stated in Answer B."
  ]
}
```
