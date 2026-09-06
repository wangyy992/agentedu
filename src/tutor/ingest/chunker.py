"""按结构切块。

先按 Markdown 标题分小节(保留层级路径),小节内再按段落聚合到目标长度。
保留 section 路径很关键——概念抽取和检索都靠它把零散 chunk 归拢回知识点。
"""
from __future__ import annotations

import re

from ..schemas import Chunk, Material

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

TARGET_CHARS = 700
MAX_CHARS = 1100
OVERLAP_CHARS = 80


def split_sections(text: str) -> list[tuple[str, str]]:
    """返回 [(标题路径, 正文)]。没有标题的文档整体作为一节。"""
    lines = text.split("\n")
    stack: list[str] = []
    sections: list[tuple[str, str]] = []
    buf: list[str] = []
    current = ""

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((current, body))
        buf.clear()

    for line in lines:
        m = HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack[:] = stack[: level - 1]
            stack.append(m.group(2))
            current = " > ".join(stack)
        else:
            buf.append(line)
    flush()

    if not sections:
        return [("", text.strip())] if text.strip() else []
    return sections


def chunk_material(material_id: str, title: str, text: str, source: str = "") -> Material:
    chunks: list[Chunk] = []
    order = 0
    for section, body in split_sections(text):
        for piece in _pack_paragraphs(body):
            order += 1
            chunks.append(
                Chunk(id=f"c{order:03d}", section=section or title, text=piece, order=order)
            )
    return Material(id=material_id, title=title, source=source, chunks=chunks)


def _pack_paragraphs(body: str) -> list[str]:
    """把段落攒到 TARGET_CHARS 左右;超长段落再按句子硬切,并带一点重叠。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out: list[str] = []
    buf = ""
    for para in paragraphs:
        for piece in _split_long(para):
            if not buf:
                buf = piece
            elif len(buf) + len(piece) + 1 <= TARGET_CHARS:
                buf = f"{buf}\n{piece}"
            else:
                out.append(buf)
                buf = piece
    if buf:
        out.append(buf)
    return out


def _split_long(para: str) -> list[str]:
    if len(para) <= MAX_CHARS:
        return [para]
    sentences = re.split(r"(?<=[。!?.!?;;])\s*", para)
    out: list[str] = []
    buf = ""
    for s in sentences:
        if not s:
            continue
        if len(buf) + len(s) <= TARGET_CHARS:
            buf += s
        else:
            if buf:
                out.append(buf)
                buf = buf[-OVERLAP_CHARS:] + s  # 少量重叠,避免语义被切断
            else:
                out.append(s[:MAX_CHARS])
                buf = s[MAX_CHARS:]
    if buf:
        out.append(buf)
    return out
