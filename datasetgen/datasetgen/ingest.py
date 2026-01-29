import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


SUPPORTED_EXTS = {".md", ".txt", ".rst", ".html", ".htm", ".pdf"}

HTML_TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\\1>", re.IGNORECASE | re.DOTALL)


@dataclass
class Doc:
    doc_id: str
    title: str
    source_path: str
    license: str
    text: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _strip_html(text: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        text = SCRIPT_STYLE_RE.sub(" ", text)
        text = HTML_TAG_RE.sub(" ", text)
        return " ".join(text.split())

    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        print("warning: pypdf not installed; skipping PDF. Install with `pip install pypdf`.")
        return ""

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _normalize_text(text: str) -> str:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip() + "\n"


def _doc_id_from_path(path: Path) -> str:
    rel = path.as_posix()
    return rel.replace("/", "__").replace(" ", "_")


def ingest_paths(paths: Iterable[str], license_str: str = "unknown") -> List[Doc]:
    docs: List[Doc] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for file_path in p.rglob("*"):
                if file_path.suffix.lower() in SUPPORTED_EXTS and file_path.is_file():
                    docs.append(_ingest_file(file_path, license_str))
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            docs.append(_ingest_file(p, license_str))
    return docs


def _ingest_file(path: Path, license_str: str) -> Doc:
    if path.suffix.lower() == ".pdf":
        raw = _read_pdf(path)
    else:
        raw = _read_text(path)
        if path.suffix.lower() in {".html", ".htm"}:
            raw = _strip_html(raw)
    text = _normalize_text(raw)
    title = path.stem
    doc_id = _doc_id_from_path(path)
    return Doc(doc_id=doc_id, title=title, source_path=str(path), license=license_str, text=text)


def write_corpus(docs: List[Doc], out_path: str) -> None:
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc.__dict__, ensure_ascii=True) + "\n")


def load_corpus(path: str) -> List[Dict[str, str]]:
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items
