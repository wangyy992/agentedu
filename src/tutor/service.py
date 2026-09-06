"""把 LLM、检索、存储、编排组装成应用层门面。

CLI 和 HTTP API 都调它,保证两个入口行为一致。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .agent.orchestrator import Course, Tutor, prepare_course
from .ingest.chunker import chunk_material
from .ingest.loader import load_text
from .llm import LLMClient, build_llm
from .memory.store import Store
from .schemas import Material, SessionState


class TutorService:
    def __init__(self, llm: LLMClient | None = None, store: Store | None = None) -> None:
        self.llm = llm or build_llm()
        self.store = store or Store()
        self._courses: dict[str, Course] = {}   # 进程内缓存,避免重复建索引

    # -- 材料 -----------------------------------------------------------
    def ingest_path(self, path: str | Path) -> Course:
        title, text = load_text(path)
        return self.ingest_text(title, text, source=str(path))

    def ingest_text(self, title: str, text: str, source: str = "") -> Course:
        """材料 id 用内容哈希:同一份材料重复导入不会重复调用模型。"""
        material_id = "m" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

        cached = self.store.load_material(material_id)
        if cached and cached[1] and cached[2]:
            material, graph, path_ = cached
            course = Course(material, graph, path_)
            self._courses[material_id] = course
            return course

        material: Material = chunk_material(material_id, title, text, source=source)
        course = prepare_course(self.llm, material)
        self.store.save_material(material, course.graph, course.path)
        self._courses[material_id] = course
        return course

    def get_course(self, material_id: str) -> Course:
        if material_id in self._courses:
            return self._courses[material_id]
        loaded = self.store.load_material(material_id)
        if not loaded or not loaded[1] or not loaded[2]:
            raise KeyError(f"材料 {material_id} 不存在或尚未完成预处理")
        material, graph, path_ = loaded
        course = Course(material, graph, path_)
        self._courses[material_id] = course
        return course

    # -- 会话 -----------------------------------------------------------
    def start_session(self, material_id: str) -> Tutor:
        tutor = Tutor(self.llm, self.get_course(material_id))
        self.store.save_session(tutor.session)
        return tutor

    def resume(self, session_id: str) -> Tutor:
        state: SessionState | None = self.store.load_session(session_id)
        if state is None:
            raise KeyError(f"会话 {session_id} 不存在")
        return Tutor(self.llm, self.get_course(state.material_id), session=state)

    def persist(self, tutor: Tutor) -> None:
        self.store.save_session(tutor.session)

    def persist_turn(self, tutor: Tutor, turn) -> None:
        self.store.record_attempt(tutor.session.session_id, turn)
        self.store.save_session(tutor.session)
