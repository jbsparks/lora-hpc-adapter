import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def split_train_eval(items: List[Dict[str, str]], eval_ratio: float = 0.2) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    n_eval = max(1, int(len(items) * eval_ratio)) if items else 0
    eval_items = items[:n_eval]
    train_items = items[n_eval:]
    return train_items, eval_items


def write_jsonl(items: Iterable[Dict[str, str]], out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


def write_manifest(items: Iterable[Dict[str, str]], out_path: str) -> None:
    manifest: List[Dict[str, str]] = []
    for item in items:
        manifest.append({
            "instruction": item.get("instruction", ""),
            "citations": item.get("citations", []),
            "topic": item.get("topic", ""),
            "difficulty": item.get("difficulty", ""),
        })
    write_jsonl(manifest, out_path)
