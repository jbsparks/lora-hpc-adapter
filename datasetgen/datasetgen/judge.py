import json
import re
from typing import Dict, List


PLACEHOLDER_RE = re.compile(r"<partition>|<account>")
FORBID_PARTITION = re.compile(r"\b(-p|--partition)\s+\S+")
FORBID_ACCOUNT = re.compile(r"\b(-A|--account)\s+\S+")


def deterministic_validators(item: Dict[str, str]) -> List[str]:
    notes: List[str] = []
    text = item.get("output", "")
    instr = item.get("instruction", "")
    topic = item.get("topic", "")

    if "partition" in text or "account" in text:
        if not PLACEHOLDER_RE.search(text) and (FORBID_PARTITION.search(text) or FORBID_ACCOUNT.search(text)):
            notes.append("Hard-coded partition/account flags found; use placeholders.")

    if topic == "slurm.interactive" and "srun --pty" not in text:
        notes.append("Interactive topic missing 'srun --pty'.")

    if "gpu" in instr.lower() and "--gres=gpu" not in text and "--gpus" not in text:
        notes.append("GPU question missing --gres=gpu or --gpus.")

    if len(text.split()) > 220:
        notes.append("Answer too long; keep concise.")

    return notes


def judge_stub(question: str, answer: str) -> Dict[str, str]:
    verdict = {
        "keep": True,
        "grounded": True,
        "notes": "stub-judge",
        "fix": "",
    }
    return verdict


def apply_repair(answer: str, fix: str) -> str:
    if not fix:
        return answer
    return fix
