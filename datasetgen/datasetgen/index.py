import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class IndexEntry:
    chunk_id: str
    doc_id: str
    text: str
    keywords: List[str]


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in text.split() if tok.strip()]


def build_bm25_index(chunks: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    doc_freq: Dict[str, int] = defaultdict(int)
    term_freqs: Dict[str, Counter] = {}
    doc_len: Dict[str, int] = {}

    for chunk in chunks:
        cid = chunk["chunk_id"]
        tokens = _tokenize(chunk["text"])
        tf = Counter(tokens)
        term_freqs[cid] = tf
        doc_len[cid] = len(tokens)
        for term in tf:
            doc_freq[term] += 1

    n_docs = len(term_freqs)
    avgdl = sum(doc_len.values()) / max(1, n_docs)
    k1 = 1.5
    b = 0.75

    index: Dict[str, Dict[str, float]] = {}
    for cid, tf in term_freqs.items():
        scores: Dict[str, float] = {}
        for term, freq in tf.items():
            df = doc_freq[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len[cid] / avgdl)
            score = idf * (freq * (k1 + 1) / denom)
            scores[term] = score
        index[cid] = scores
    return index


def save_index(chunks: List[Dict[str, str]], index: Dict[str, Dict[str, float]], out_dir: str) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    with (p / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=True) + "\n")
    with (p / "bm25.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=True, sort_keys=True)


def load_index(index_dir: str) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, float]]]:
    p = Path(index_dir)
    chunks: List[Dict[str, str]] = []
    with (p / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    with (p / "bm25.json").open("r", encoding="utf-8") as f:
        index = json.load(f)
    return chunks, index


def _score_query(query: str, index: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    tokens = _tokenize(query)
    scores: Dict[str, float] = defaultdict(float)
    for cid, term_scores in index.items():
        for term in tokens:
            if term in term_scores:
                scores[cid] += term_scores[term]
    return scores


def retrieve(query: str, chunks: List[Dict[str, str]], index: Dict[str, Dict[str, float]], top_k: int = 3) -> List[Dict[str, str]]:
    scores = _score_query(query, index)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    cid_to_chunk = {c["chunk_id"]: c for c in chunks}
    results: List[Dict[str, str]] = []
    for cid, _ in ranked[:top_k]:
        if cid in cid_to_chunk:
            results.append(cid_to_chunk[cid])
    if not results and chunks:
        results = chunks[:top_k]
    return results
