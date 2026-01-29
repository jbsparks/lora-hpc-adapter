import re
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    section: str
    text: str
    keywords: List[str]


HEADER_RE = re.compile(r"^(#+)\s+(.*)")
COMMAND_RE = re.compile(r"\b(srun|salloc|sbatch|sinfo|squeue|sacct|scontrol)\b")


def _extract_section(lines: List[str]) -> str:
    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            return m.group(2).strip()
    return ""


def _extract_keywords(text: str) -> List[str]:
    kws = sorted(set(COMMAND_RE.findall(text)))
    return kws


def chunk_text(doc: Dict[str, str], max_chars: int = 800, overlap: int = 120) -> List[Chunk]:
    text = doc["text"]
    lines = text.split("\n")
    chunks: List[Chunk] = []
    buf: List[str] = []
    size = 0
    section = _extract_section(lines)
    idx = 0

    def flush(buffer: List[str], idx: int) -> None:
        if not buffer:
            return
        chunk_text = "\n".join(buffer).strip()
        if not chunk_text:
            return
        chunk_id = f"{doc['doc_id']}__{idx:04d}"
        keywords = _extract_keywords(chunk_text)
        chunks.append(Chunk(doc_id=doc["doc_id"], chunk_id=chunk_id, section=section, text=chunk_text, keywords=keywords))

    for line in lines:
        if size + len(line) + 1 > max_chars:
            flush(buf, idx)
            idx += 1
            if overlap > 0 and buf:
                overlap_text = "\n".join(buf)[-overlap:]
                buf = [overlap_text]
                size = len(overlap_text)
            else:
                buf = []
                size = 0
        buf.append(line)
        size += len(line) + 1

    flush(buf, idx)
    return chunks


def chunk_corpus(corpus: Iterable[Dict[str, str]], max_chars: int = 800, overlap: int = 120) -> List[Chunk]:
    out: List[Chunk] = []
    for doc in corpus:
        out.extend(chunk_text(doc, max_chars=max_chars, overlap=overlap))
    return out
