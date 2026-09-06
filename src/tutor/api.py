"""HTTP 接口 + 静态网页。

    uvicorn tutor.api:app --reload
    或 python -m tutor.cli serve

状态放在服务端(SQLite),前端只拿 session_id,刷新页面不会丢进度。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import guardrails as g
from .examples import example_path, list_examples
from .llm import FakeLLM, build_llm
from .schemas import ActionType
from .service import TutorService

app = FastAPI(title="AI 自适应辅导 Agent", version="0.1.0")
service = TutorService()

# 超出每日预算时切过去的离线兜底模型:演示站宁可效果变差,也不要看起来挂了
_fallback_llm = FakeLLM()

WEB_DIR = Path(__file__).resolve().parent / "web"


# --- 请求体 -------------------------------------------------------------
class IngestBody(BaseModel):
    title: str = "未命名材料"
    text: str = ""
    path: str = ""       # 服务器本地路径。演示模式下被拒绝(见 guardrails)
    example: str = ""    # 内置示例名,如 "gradient_descent"


class AnswerBody(BaseModel):
    answer: str


class AskBody(BaseModel):
    question: str


# --- 序列化 -------------------------------------------------------------
def _turn_payload(tutor, turn) -> dict[str, Any]:
    concept = tutor.course.concepts.get(turn.concept_id) if turn.concept_id else None
    state = tutor.session.concepts.get(turn.concept_id) if turn.concept_id else None
    return {
        "step": turn.step,
        "action": turn.action.value,
        "concept": {"id": concept.id, "name": concept.name} if concept else None,
        "reason": turn.reason,
        "content": turn.content,
        "citations": turn.citations,
        "item": (
            {
                "id": turn.item.id,
                "kind": turn.item.kind.value,
                "difficulty": turn.item.difficulty,
                "stem": turn.item.stem,
                "options": turn.item.options,
                "source_chunk_ids": turn.item.source_chunk_ids,
            }
            if turn.item else None
        ),
        "mastery": round(state.p_known, 3) if state else None,
        "finished": tutor.session.finished,
    }


def _progress(tutor) -> list[dict[str, Any]]:
    names = tutor.course.names()
    return [
        {
            "id": cid,
            "name": names.get(cid, cid),
            "mastery": round(tutor.session.concepts[cid].p_known, 3),
            "attempts": tutor.session.concepts[cid].attempts,
            "current": cid == tutor.session.current_concept_id(),
        }
        for cid in tutor.session.path.ordered_concept_ids
        if cid in tutor.session.concepts
    ]


# --- 护栏 ---------------------------------------------------------------
def _guard(request: Request, limiter: g.SlidingWindow | None = None):
    """每个会花钱的路由都先过这里:限流 + 按预算决定用哪个模型。

    返回 (生效的模型, 调用前的计数),配合 _spend() 记账。
    """
    key = g.client_key(request)
    try:
        g.request_limiter.check(key)
        if limiter is not None:
            limiter.check(key)
    except g.RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc),
                            headers={"Retry-After": str(exc.retry_after)}) from exc

    # 预算用尽 -> 降级到离线模型。请求照常成功,只是内容质量下降:
    # 演示站宁可效果变差,也不要看起来挂了。
    llm = service.set_degraded(g.budget.exhausted(), _fallback_llm)
    return llm, llm.usage.calls


def _spend(llm, before: int) -> None:
    """记录本次请求真实消耗的模型调用数。

    必须用 _guard 返回的那个 llm 对象,而不是重新读 service.llm——
    并发请求下后者可能已被另一个请求换成了兜底模型,差值会算错。
    """
    g.budget.record(max(0, llm.usage.calls - before))


# --- 路由 ---------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "llm": type(service.llm).__name__,
        "demo_mode": g.LIMITS.demo_mode,
        "budget": g.budget.snapshot(),
        "examples": list_examples(),
        "max_material_chars": g.LIMITS.max_material_chars,
    }


@app.get("/api/materials")
def list_materials() -> list[dict[str, Any]]:
    return service.store.list_materials()


@app.post("/api/materials")
def ingest(body: IngestBody, request: Request) -> dict[str, Any]:
    """导入材料:理解 -> 建知识图谱 -> 规划路径。同一份材料只会处理一次。

    三种入口,优先级 example > path > text。path 在演示模式下会被拒绝——
    否则任何人都能让服务端读取任意文件并把内容回显出来。
    """
    llm, before = _guard(request, g.ingest_limiter)
    try:
        if body.example:
            course = service.ingest_path(example_path(body.example))
        elif body.path:
            course = service.ingest_path(g.resolve_material_path(body.path))
        else:
            text = g.validate_material_text(body.text)
            course = service.ingest_text(body.title or "粘贴的材料", text)
    except (g.MaterialRejected, KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip("'")) from exc
    finally:
        _spend(llm, before)

    return {
        "material_id": course.material.id,
        "title": course.material.title,
        "chunks": len(course.material.chunks),
        "concepts": [
            {
                "id": c.id, "name": c.name, "summary": c.summary,
                "difficulty": c.difficulty, "prerequisites": c.prerequisites,
            }
            for c in course.graph.concepts
        ],
        "path": course.path.ordered_concept_ids,
        "rationale": course.path.rationale,
    }


@app.post("/api/sessions")
def start_session(body: dict, request: Request) -> dict[str, Any]:
    _guard(request)
    material_id = body.get("material_id", "")
    try:
        tutor = service.start_session(material_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"session_id": tutor.session.session_id, "progress": _progress(tutor)}


def _load(session_id: str):
    try:
        return service.resume(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/next")
def next_step(session_id: str, request: Request) -> dict[str, Any]:
    llm, before = _guard(request)
    tutor = _load(session_id)
    turn = tutor.next_step()
    _spend(llm, before)
    service.persist(tutor)
    payload = _turn_payload(tutor, turn)
    payload["progress"] = _progress(tutor)
    return payload


@app.post("/api/sessions/{session_id}/answer")
def submit_answer(session_id: str, body: AnswerBody, request: Request) -> dict[str, Any]:
    llm, before = _guard(request)
    tutor = _load(session_id)
    if tutor.session.pending is None:
        raise HTTPException(status_code=409, detail="当前没有待作答的题目")
    turn = tutor.submit_answer(body.answer[: g.LIMITS.max_material_chars])
    _spend(llm, before)
    service.persist_turn(tutor, turn)
    grade = turn.grade
    return {
        "score": grade.score,
        "verdict": grade.verdict.value,
        "feedback": grade.feedback,
        "missing_points": grade.missing_points,
        "misconception_tags": grade.misconception_tags,
        "answer_key": turn.item.answer_key if turn.item else "",
        "rationale": turn.item.rationale if turn.item else "",
        "hint": turn.item.hint if turn.item else "",
        "mastery": round(tutor.session.concepts[turn.concept_id].p_known, 3),
        "progress": _progress(tutor),
    }


@app.post("/api/sessions/{session_id}/ask")
def ask(session_id: str, body: AskBody, request: Request) -> dict[str, Any]:
    llm, before = _guard(request)
    question = (body.question or "").strip()[:500]
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    tutor = _load(session_id)
    result = tutor.ask(question)
    _spend(llm, before)
    return result.model_dump()


@app.get("/api/sessions/{session_id}/report")
def report(session_id: str) -> dict[str, Any]:
    tutor = _load(session_id)
    payload = tutor.report().model_dump()
    payload["wrong_items"] = service.store.wrong_items(session_id)
    return payload


@app.get("/api/sessions/{session_id}/state")
def state(session_id: str) -> dict[str, Any]:
    """调试用:把 agent 的内部状态整个吐出来,方便观察自适应是怎么决策的。"""
    tutor = _load(session_id)
    return {
        "step": tutor.session.step,
        "cursor": tutor.session.cursor,
        "finished": tutor.session.finished,
        "review_queue": tutor.session.review_queue,
        "concepts": {
            cid: s.model_dump() for cid, s in tutor.session.concepts.items()
        },
        "history": [
            {"step": t.step, "action": t.action.value, "concept_id": t.concept_id,
             "score": t.grade.score if t.grade else None, "reason": t.reason}
            for t in tutor.session.history
        ],
        "usage": service.llm.usage.as_dict(),
    }


# --- 静态页面 -----------------------------------------------------------
if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))
