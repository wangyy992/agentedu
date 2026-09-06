"""SQLite 持久化:材料、知识图谱、会话状态、逐题作答流水。

刻意把 SessionState 整体以 JSON 存一列,而不是拆成十几张表:
这是一个状态机快照,读写永远是整体的,拆表只会带来同步负担。
但 attempts 单独落一张明细表——它是**分析用**的数据(错题回溯、误区统计、
后续做 offline eval 或训练难度模型),需要能按题、按概念、按时间查询。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..config import DB_PATH
from ..schemas import ConceptGraph, LearningPath, Material, SessionState, Turn

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    source     TEXT,
    chunks     TEXT NOT NULL,          -- JSON: Material
    graph      TEXT,                   -- JSON: ConceptGraph
    path       TEXT,                   -- JSON: LearningPath
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(id),
    state       TEXT NOT NULL,         -- JSON: SessionState
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    step        INTEGER NOT NULL,
    concept_id  TEXT,
    action      TEXT,
    item_id     TEXT,
    difficulty  INTEGER,
    kind        TEXT,
    stem        TEXT,
    answer      TEXT,
    score       REAL,
    verdict     TEXT,
    misconceptions TEXT,               -- JSON: list[str]
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_attempts_session ON attempts(session_id, step);
CREATE INDEX IF NOT EXISTS idx_attempts_concept ON attempts(concept_id);
"""


@contextmanager
def connect(db_path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


class Store:
    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = db_path

    # -- 材料 -----------------------------------------------------------
    def save_material(
        self, material: Material, graph: ConceptGraph | None = None,
        path: LearningPath | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO materials (id,title,source,chunks,graph,path) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET title=excluded.title, chunks=excluded.chunks, "
                "graph=excluded.graph, path=excluded.path",
                (
                    material.id, material.title, material.source,
                    material.model_dump_json(),
                    graph.model_dump_json() if graph else None,
                    path.model_dump_json() if path else None,
                ),
            )

    def load_material(self, material_id: str
                      ) -> tuple[Material, ConceptGraph | None, LearningPath | None] | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM materials WHERE id=?", (material_id,)).fetchone()
        if row is None:
            return None
        return (
            Material.model_validate_json(row["chunks"]),
            ConceptGraph.model_validate_json(row["graph"]) if row["graph"] else None,
            LearningPath.model_validate_json(row["path"]) if row["path"] else None,
        )

    def list_materials(self) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id,title,source,created_at FROM materials ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # -- 会话 -----------------------------------------------------------
    def save_session(self, session: SessionState) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (id,material_id,state,updated_at) "
                "VALUES (?,?,?,CURRENT_TIMESTAMP) "
                "ON CONFLICT(id) DO UPDATE SET state=excluded.state, updated_at=CURRENT_TIMESTAMP",
                (session.session_id, session.material_id, session.model_dump_json()),
            )

    def load_session(self, session_id: str) -> SessionState | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT state FROM sessions WHERE id=?", (session_id,)).fetchone()
        return SessionState.model_validate_json(row["state"]) if row else None

    # -- 作答明细 -------------------------------------------------------
    def record_attempt(self, session_id: str, turn: Turn) -> None:
        if turn.item is None or turn.grade is None:
            return
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO attempts (session_id,step,concept_id,action,item_id,difficulty,"
                "kind,stem,answer,score,verdict,misconceptions) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    session_id, turn.step, turn.concept_id, turn.action.value,
                    turn.item.id, turn.item.difficulty, turn.item.kind.value, turn.item.stem,
                    turn.student_answer, turn.grade.score, turn.grade.verdict.value,
                    json.dumps(turn.grade.misconception_tags, ensure_ascii=False),
                ),
            )

    def wrong_items(self, session_id: str, limit: int = 20) -> list[dict]:
        """错题本:分数低于 0.8 的作答,按时间倒序。"""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT step,concept_id,stem,answer,score,verdict,misconceptions "
                "FROM attempts WHERE session_id=? AND score < 0.8 ORDER BY step DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["misconceptions"] = json.loads(d["misconceptions"] or "[]")
            out.append(d)
        return out

    def misconception_stats(self, material_id: str | None = None) -> list[dict]:
        """跨会话的误区聚合——教研视角:这份材料上大家普遍卡在哪。"""
        sql = (
            "SELECT a.concept_id, a.misconceptions FROM attempts a "
            "JOIN sessions s ON s.id = a.session_id WHERE a.score < 0.8"
        )
        args: tuple = ()
        if material_id:
            sql += " AND s.material_id = ?"
            args = (material_id,)
        with connect(self.db_path) as conn:
            rows = conn.execute(sql, args).fetchall()

        tally: dict[str, int] = {}
        for r in rows:
            for tag in json.loads(r["misconceptions"] or "[]"):
                tally[tag] = tally.get(tag, 0) + 1
        return [
            {"tag": t, "count": c} for t, c in sorted(tally.items(), key=lambda kv: -kv[1])
        ]
