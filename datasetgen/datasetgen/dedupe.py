import hashlib
from typing import Dict, Iterable, List, Set


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def exact_dedupe(items: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: Set[str] = set()
    out: List[Dict[str, str]] = []
    for item in items:
        key = hashlib.sha256((_norm(item["instruction"]) + "||" + _norm(item["output"])).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def token_jaccard_dedupe(items: Iterable[Dict[str, str]], threshold: float = 0.92) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    signatures: List[Set[str]] = []

    for item in items:
        toks = set(_norm(item["instruction"]).split()) | set(_norm(item["output"]).split())
        duplicate = False
        for sig in signatures:
            if not sig or not toks:
                continue
            j = len(sig & toks) / len(sig | toks)
            if j >= threshold:
                duplicate = True
                break
        if not duplicate:
            signatures.append(toks)
            out.append(item)
    return out
