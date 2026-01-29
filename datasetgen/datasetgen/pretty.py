import json
from pathlib import Path
from typing import List


def pretty_print_jsonl(in_path: str, out_path: str = "", mode: str = "json", limit: int = 0) -> None:
    p = Path(in_path)
    out = Path(out_path) if out_path else None
    items: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if limit:
        items = items[:limit]

    if mode == "text":
        chunks = []
        for i, item in enumerate(items, start=1):
            chunks.append(f"=== item {i} ===")
            chunks.append(f"instruction: {item.get('instruction','')}")
            chunks.append(f"topic: {item.get('topic','')}")
            chunks.append(f"difficulty: {item.get('difficulty','')}")
            chunks.append(f"citations: {item.get('citations',[])}")
            chunks.append("output:")
            chunks.append(item.get("output", ""))
            chunks.append("")
        text = "\n".join(chunks)
    else:
        text = json.dumps(items, ensure_ascii=True, indent=2, sort_keys=False)
    if out:
        out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
