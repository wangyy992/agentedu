"""把各种格式的学习材料读成纯文本。

txt / md / html 无依赖;pdf 走 pypdf(可选依赖,没装时给出明确提示)。
"""
from __future__ import annotations

import html
import re
from pathlib import Path


def load_text(path: str | Path) -> tuple[str, str]:
    """返回 (标题, 正文)。标题优先取一级标题,否则用文件名。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到材料文件: {p}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        text = _load_pdf(p)
    elif suffix in (".html", ".htm"):
        text = _strip_html(p.read_text(encoding="utf-8", errors="replace"))
    else:
        text = p.read_text(encoding="utf-8", errors="replace")

    text = _normalize(text)
    title = _guess_title(text) or p.stem
    return title, text


def _load_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise RuntimeError("解析 PDF 需要 pypdf,请先 `pip install pypdf`") from exc

    reader = PdfReader(str(p))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        body = page.extract_text() or ""
        if body.strip():
            pages.append(f"## 第 {i} 页\n{body}")
    if not pages:
        raise RuntimeError(f"{p.name} 未能提取到文本,可能是扫描件,需要先做 OCR")
    return "\n\n".join(pages)


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)>", "\n", raw)
    raw = re.sub(r"(?i)<h([1-6])[^>]*>", lambda m: "\n" + "#" * int(m.group(1)) + " ", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return html.unescape(raw)


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_title(text: str) -> str:
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line and not line.startswith("#"):
            return line[:60]
    return ""
