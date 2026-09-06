"""命令行入口。

    python -m tutor.cli ingest gradient_descent
    python -m tutor.cli tutor  gradient_descent          # 也可以直接给一个文件路径
    python -m tutor.cli ask    <material_id> "学习率太大会怎样"
    python -m tutor.cli eval   gradient_descent --ability 0.5
    python -m tutor.cli serve

加 --fake 可在无 API Key 的情况下跑完整流程(与 CI 走的是同一条路径)。
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from .agent.orchestrator import Tutor
from .examples import list_examples, resolve
from .llm import build_llm
from .memory.store import Store
from .schemas import ActionType, ItemKind, Turn
from .service import TutorService

W = 78


def _hr(char: str = "─") -> None:
    print(char * W)


def _wrap(text: str, indent: str = "") -> str:
    out = []
    for para in (text or "").split("\n"):
        out.append(textwrap.fill(para, width=W, initial_indent=indent, subsequent_indent=indent)
                   if para.strip() else "")
    return "\n".join(out)


def _service(args) -> TutorService:
    llm = build_llm(force_fake=True) if args.fake else build_llm()
    return TutorService(llm=llm, store=Store())


# --- 子命令 -------------------------------------------------------------
def cmd_ingest(args) -> int:
    svc = _service(args)
    course = svc.ingest_path(resolve(args.path))
    print(f"材料 id: {course.material.id}  片段数: {len(course.material.chunks)}")
    _hr()
    print("知识图谱:")
    for c in course.graph.concepts:
        prereq = f"  ← 依赖 {', '.join(c.prerequisites)}" if c.prerequisites else ""
        print(f"  [{c.id}] 难度{c.difficulty} {c.name}{prereq}")
        print(_wrap(c.summary, indent="        "))
    _hr()
    print("学习路径:", " → ".join(
        course.concept(cid).name for cid in course.path.ordered_concept_ids))
    print(_wrap(course.path.rationale))
    print(f"\n模型用量: {svc.llm.usage.as_dict()}")
    return 0


def cmd_tutor(args) -> int:
    svc = _service(args)
    course = svc.ingest_path(resolve(args.path))
    tutor = svc.start_session(course.material.id)
    print(f"会话 {tutor.session.session_id} 已开始。输入 :q 退出,:? 后跟问题可随时提问。")
    _hr("═")

    while True:
        turn = tutor.next_step()
        if turn.action is ActionType.COMPLETE:
            _hr("═")
            print("学习完成\n")
            print(_wrap(turn.content))
            break

        _render_turn(turn, course)
        if turn.item is None:
            svc.persist(tutor)
            continue

        answer = _read_answer(tutor)
        if answer is None:
            print("\n已保存进度,下次可用 resume 继续:", tutor.session.session_id)
            svc.persist(tutor)
            return 0

        done = tutor.submit_answer(answer)
        _render_grade(done, tutor)
        svc.persist_turn(tutor, done)

    report = tutor.report()
    _render_report(report)
    print(f"\n模型用量: {svc.llm.usage.as_dict()}")
    return 0


def cmd_ask(args) -> int:
    svc = _service(args)
    course = svc.get_course(args.material_id)
    tutor = Tutor(svc.llm, course)
    result = tutor.ask(args.question)
    print(_wrap(result.answer))
    _hr()
    print("检索关键词:", result.used_searches)
    print("已核验引用:", result.citations or "(无——回答未能溯源到材料)")
    return 0


def cmd_eval(args) -> int:
    from .evaluation.simulate import run_simulation, format_result

    svc = _service(args)
    course = svc.ingest_path(resolve(args.path))
    result = run_simulation(svc.llm, course, ability=args.ability, seed=args.seed,
                            max_steps=args.max_steps)
    print(format_result(result))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("tutor.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# --- 渲染 ---------------------------------------------------------------
LABEL = {
    ActionType.TEACH: "讲解",
    ActionType.PRACTICE: "练习",
    ActionType.REMEDIATE: "补救",
    ActionType.REVIEW: "复习",
}


def _render_turn(turn: Turn, course) -> None:
    name = course.concept(turn.concept_id).name if turn.concept_id else ""
    _hr()
    print(f"[{LABEL.get(turn.action, turn.action.value)}] {name}")
    print(f"  ↳ 策略:{turn.reason}")
    _hr()
    if turn.content:
        print(_wrap(turn.content))
        print()
    if turn.item:
        print(f"【第 {turn.item.id} 题 · 难度 {turn.item.difficulty}/5 · {turn.item.kind.value}】")
        print(_wrap(turn.item.stem))
        for opt in turn.item.options:
            print(f"    {opt}")
        print(f"  (依据片段:{', '.join(turn.item.source_chunk_ids)})")


def _read_answer(tutor: Tutor) -> str | None:
    """读一行作答;支持中途提问和退出。"""
    while True:
        try:
            raw = input("\n你的回答 > ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw in (":q", ":quit"):
            return None
        if raw.startswith(":?"):
            question = raw[2:].strip()
            if question:
                result = tutor.ask(question)
                _hr("·")
                print(_wrap(result.answer))
                print("引用:", result.citations or "(未溯源)")
                _hr("·")
            continue
        if raw:
            return raw
        print("(直接回车会算作未作答;不确定也可以写下思路)")


def _render_grade(turn: Turn, tutor: Tutor) -> None:
    grade = turn.grade
    mark = {"correct": "✓ 正确", "partial": "~ 部分正确", "incorrect": "✗ 错误"}[grade.verdict.value]
    state = tutor.session.concepts[turn.concept_id]
    _hr("·")
    print(f"{mark}  得分 {grade.score:.0%}   掌握度 {state.p_known:.0%} "
          f"{_bar(state.p_known)}")
    print(_wrap(grade.feedback))
    if grade.missing_points:
        print("  还缺:" + "、".join(grade.missing_points))
    if grade.misconception_tags:
        print("  记录误区:" + "、".join(grade.misconception_tags))
    if turn.item and grade.verdict.value != "correct":
        print("  提示:" + turn.item.hint)
        if turn.item.kind is not ItemKind.SHORT:
            print(f"  参考答案:{turn.item.answer_key}")


def _bar(x: float, width: int = 20) -> str:
    filled = int(round(x * width))
    return "▊" * filled + "·" * (width - filled)


def _render_report(report) -> None:
    _hr("═")
    print(f"整体掌握度 {report.overall:.0%} {_bar(report.overall)}")
    print("已掌握:", "、".join(report.mastered) or "无")
    print("待巩固:", "、".join(report.shaky) or "无")
    if report.not_started:
        print("未开始:", "、".join(report.not_started))
    if report.top_misconceptions:
        print("高频误区:")
        for m in report.top_misconceptions:
            print(f"  · {m['tag']} ×{m['count']}")


# --- 入口 ---------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tutor", description="AI 自适应辅导 Agent")
    parser.add_argument("--fake", action="store_true", help="使用离线假模型(无需 API Key)")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印 agent 内部日志")
    sub = parser.add_subparsers(dest="command", required=True)

    examples_help = "材料文件路径,或内置示例名(可用:" + "、".join(list_examples()) + ")"

    p = sub.add_parser("ingest", help="导入材料并生成知识图谱与学习路径")
    p.add_argument("path", help=examples_help)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("tutor", help="开始一次交互式辅导")
    p.add_argument("path", help=examples_help)
    p.set_defaults(func=cmd_tutor)

    p = sub.add_parser("ask", help="就已导入的材料提问")
    p.add_argument("material_id")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="用模拟学生跑一遍,评估策略层")
    p.add_argument("path", help=examples_help)
    p.add_argument("--ability", type=float, default=0.55, help="模拟学生的能力 0~1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=80)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="启动 HTTP 服务与网页界面")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except (KeyboardInterrupt, EOFError):
        print("\n已中断。")
        return 130
    except Exception as exc:  # 命令行不该甩栈给用户
        if args.verbose:
            raise
        print(f"错误:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
