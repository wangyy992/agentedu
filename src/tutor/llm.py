"""与 Claude 的所有交互都收敛在这一层。

三件事:
1. **结构化输出** —— 用 output_config.format 约束 JSON,再用 pydantic 二次校验,
   失败自动带着报错重试一次(JSON repair)。
2. **Prompt caching** —— 材料全文放在缓存断点内,任务指令放在断点外。
   同一份材料的几十次调用共享同一段缓存前缀。
3. **Tool-use 循环** —— 手写 while 循环(不依赖 beta 版 tool_runner),
   便于插桩、审计和离线测试。

`FakeLLM` 实现同一套接口,用确定性规则产出结果,使 `TUTOR_FAKE_LLM=1`
时整个 agent 无需 API Key 即可端到端运行——测试和 CI 全靠它。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import EFFORT_BY_TASK, MAX_TOKENS, MODEL, RUNTIME
from .schemas import strict_schema

log = logging.getLogger("tutor.llm")

T = TypeVar("T", bound=BaseModel)

BASE_SYSTEM = (
    "你是一位严谨的学科助教。你的所有讲解、出题和判分都必须严格基于用户提供的学习材料,"
    "不得引入材料之外的知识点。引用材料时使用 chunk 编号(形如 c012)。"
    "如果材料中没有相关依据,明确说明「材料未涉及」,不要编造。"
)


@dataclass
class Prompt:
    """一次调用的完整描述。

    text 侧(system/user)给真实模型;data 侧给离线模型做确定性推导。
    两者同源,保证离线测试覆盖的是同一条代码路径。
    """

    task: str
    system: str
    user: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    calls: int = 0

    def add(self, u: Any) -> None:
        self.calls += 1
        self.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "output_tokens", 0) or 0
        self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read,
            "cache_creation_input_tokens": self.cache_write,
        }


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: str


@dataclass
class ToolLoopResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    """agent 各模块只依赖这个协议,不直接依赖 anthropic。"""

    usage: Usage

    def structured(self, prompt: Prompt, model_cls: type[T], *, cached_context: str = "") -> T: ...

    def text(self, prompt: Prompt, *, cached_context: str = "") -> str: ...

    def tool_loop(
        self,
        prompt: Prompt,
        *,
        tools: list[dict[str, Any]],
        executor: Callable[[str, dict[str, Any]], str],
        cached_context: str = "",
        max_turns: int = 6,
    ) -> ToolLoopResult: ...


# =======================================================================
# 真实实现
# =======================================================================
class AnthropicLLM:
    def __init__(self, model: str = MODEL, client: Any = None) -> None:
        self.model = model
        self.usage = Usage()
        self.__client = client

    @property
    def _client(self) -> Any:
        """懒构造。

        构造 anthropic.Anthropic() 时若没有凭据会直接抛异常,所以不能放在
        __init__ 里——否则「导入应用」这件事本身就会因为缺 Key 而失败,
        连健康检查和静态页面都起不来。改成首次真正调用模型时才构造。
        """
        if self.__client is None:
            import anthropic  # 延迟导入:离线模式下不装 SDK 也能跑

            try:
                self.__client = anthropic.Anthropic()
            except Exception as exc:
                raise RuntimeError(
                    "无法初始化 Claude 客户端:请设置 ANTHROPIC_API_KEY,"
                    "或用离线模式(TUTOR_FAKE_LLM=1 / CLI 加 --fake)。"
                    f"原始错误:{exc}"
                ) from exc
        return self.__client

    # -- prompt 组装 ----------------------------------------------------
    def _system_blocks(self, task_system: str, cached_context: str) -> list[dict[str, Any]]:
        """缓存断点之前放不变的内容(全局 system + 材料全文),之后放任务指令。

        顺序是 tools -> system -> messages,前缀一旦变化缓存全失效,
        所以易变的任务指令必须排在 cache_control 标记的块之后。
        """
        blocks: list[dict[str, Any]] = [{"type": "text", "text": BASE_SYSTEM}]
        if cached_context:
            blocks.append(
                {
                    "type": "text",
                    "text": f"<学习材料>\n{cached_context}\n</学习材料>",
                    # 同一份材料的所有调用共享这段前缀;1h TTL 覆盖一次完整辅导会话
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            )
        blocks.append({"type": "text", "text": task_system})
        return blocks

    def _effort(self, task: str) -> str:
        return EFFORT_BY_TASK.get(task, "high")

    # -- 结构化输出 -----------------------------------------------------
    def structured(self, prompt: Prompt, model_cls: type[T], *, cached_context: str = "") -> T:
        schema = strict_schema(model_cls)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt.user}]

        last_error = ""
        last_raw = ""
        for attempt in range(2):
            if attempt == 1:
                # JSON repair:把校验错误回灌,让模型自己改
                messages = messages + [
                    {"role": "assistant", "content": last_raw or "(空)"},
                    {
                        "role": "user",
                        "content": (
                            f"上一次输出不符合 schema,错误如下:\n{last_error}\n"
                            "请只输出修正后的 JSON,不要解释。"
                        ),
                    },
                ]
            response = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self._system_blocks(prompt.system, cached_context),
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self._effort(prompt.task),
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
            self.usage.add(response.usage)
            last_raw = _first_text(response)
            try:
                return model_cls.model_validate_json(last_raw)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:1500]
                log.warning("task=%s 结构化输出校验失败(第 %d 次): %s", prompt.task, attempt + 1, last_error[:200])

        raise RuntimeError(f"task={prompt.task} 连续两次输出不符合 schema: {last_error}")

    # -- 纯文本 ---------------------------------------------------------
    def text(self, prompt: Prompt, *, cached_context: str = "") -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=self._system_blocks(prompt.system, cached_context),
            messages=[{"role": "user", "content": prompt.user}],
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort(prompt.task)},
        )
        self.usage.add(response.usage)
        return _first_text(response)

    # -- tool-use 循环 --------------------------------------------------
    def tool_loop(
        self,
        prompt: Prompt,
        *,
        tools: list[dict[str, Any]],
        executor: Callable[[str, dict[str, Any]], str],
        cached_context: str = "",
        max_turns: int = 6,
    ) -> ToolLoopResult:
        """手写 agent 循环:请求 -> 执行工具 -> 回灌结果 -> 直到 end_turn。

        并行工具调用的结果必须放在**同一条** user 消息里一次性回传,
        否则模型会逐渐不再并行调用。
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt.user}]
        calls: list[ToolCall] = []

        for _ in range(max_turns):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self._system_blocks(prompt.system, cached_context),
                messages=messages,
                tools=tools,
                thinking={"type": "adaptive"},
                output_config={"effort": self._effort(prompt.task)},
            )
            self.usage.add(response.usage)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return ToolLoopResult(text=_first_text(response), tool_calls=calls)

            results: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                args = block.input if isinstance(block.input, dict) else json.loads(block.input)
                try:
                    out = executor(block.name, args)
                    is_error = False
                except Exception as exc:  # 工具失败也要回传,不能吞掉
                    out, is_error = f"工具执行失败: {exc}", True
                calls.append(ToolCall(name=block.name, arguments=args, result=out))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": out,
                        **({"is_error": True} if is_error else {}),
                    }
                )
            messages.append({"role": "user", "content": results})

        # 兜底:超过轮数上限,让模型直接给结论
        messages.append({"role": "user", "content": "请基于已有信息直接给出最终回答,不要再调用工具。"})
        response = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=self._system_blocks(prompt.system, cached_context),
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort(prompt.task)},
        )
        self.usage.add(response.usage)
        return ToolLoopResult(text=_first_text(response), tool_calls=calls)


def _first_text(response: Any) -> str:
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


# =======================================================================
# 离线实现:确定性、无网络、无 Key
# =======================================================================
class FakeLLM:
    """用规则模拟每个任务的输出,让全流程可以离线跑通并被断言。

    它不是"能力等价"的替身,只是**接口与结构等价**——用于验证编排、状态机、
    自适应策略这些真正的工程逻辑,把模型质量的不确定性隔离在外。
    """

    def __init__(self) -> None:
        self.usage = Usage()
        self.log: list[str] = []

    def _tick(self) -> None:
        self.usage.calls += 1

    def structured(self, prompt: Prompt, model_cls: type[T], *, cached_context: str = "") -> T:
        self._tick()
        self.log.append(prompt.task)
        builder = getattr(self, f"_fake_{prompt.task}", None)
        if builder is None:
            raise NotImplementedError(f"FakeLLM 未实现 task={prompt.task}")
        return model_cls.model_validate(builder(prompt.data))

    def text(self, prompt: Prompt, *, cached_context: str = "") -> str:
        self._tick()
        self.log.append(prompt.task)
        builder = getattr(self, f"_fake_{prompt.task}", None)
        if builder is None:
            return f"[fake:{prompt.task}]"
        out = builder(prompt.data)
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)

    def tool_loop(
        self,
        prompt: Prompt,
        *,
        tools: list[dict[str, Any]],
        executor: Callable[[str, dict[str, Any]], str],
        cached_context: str = "",
        max_turns: int = 6,
    ) -> ToolLoopResult:
        """模拟一次"先检索再回答":真的调用一次 executor,保证工具链被覆盖。"""
        self._tick()
        self.log.append(prompt.task)
        query = prompt.data.get("question", "")
        result = executor("search_material", {"query": query, "top_k": 3})
        cited = re.findall(r"\[(c\d+)\]", result)[:3]
        body = f"关于「{query}」,材料中的相关内容如下:\n{result[:600]}"
        if cited:
            body += "\n\n依据:" + "、".join(cited)
        return ToolLoopResult(
            text=body,
            tool_calls=[ToolCall("search_material", {"query": query}, result)],
        )

    # -- 各任务的确定性实现 ---------------------------------------------
    @staticmethod
    def _fake_extract_concepts(data: dict[str, Any]) -> dict[str, Any]:
        """按小节切概念,并让第 i 个概念依赖第 i-1 个,形成一条链。"""
        sections: list[dict[str, Any]] = data.get("sections", [])
        concepts = []
        for i, sec in enumerate(sections):
            cid = f"k{i + 1:02d}"
            concepts.append(
                {
                    "id": cid,
                    "name": sec["title"].split(" > ")[-1],
                    "summary": sec["text"][:80],
                    "key_points": [s for s in re.split(r"[。;\n]", sec["text"]) if s.strip()][:3],
                    "prerequisites": [f"k{i:02d}"] if i > 0 else [],
                    "evidence_chunk_ids": sec["chunk_ids"][:3],
                    "difficulty": min(5, 1 + i % 4),
                }
            )
        return {"material_id": data.get("material_id", ""), "concepts": concepts}

    @staticmethod
    def _fake_explain(data: dict[str, Any]) -> str:
        name = data.get("concept_name", "该概念")
        evidence = data.get("evidence", "")
        return f"【{name}】\n{evidence[:400]}\n\n要点:{'、'.join(data.get('key_points', [])) or '见上'}"

    @staticmethod
    def _fake_write_item(data: dict[str, Any]) -> dict[str, Any]:
        """难度 <=2 出选择题,否则出简答题;答案取材料里的关键词。"""
        name = data.get("concept_name", "概念")
        difficulty = int(data.get("difficulty", 2))
        chunk_ids = data.get("chunk_ids", [])
        keyword = (data.get("key_points") or [name])[0][:20]
        if difficulty <= 2:
            return {
                "kind": "mcq",
                "difficulty": difficulty,
                "stem": f"关于「{name}」,下列说法正确的是?",
                "options": [f"A. {keyword}", "B. 与材料描述相反的说法", "C. 无关项", "D. 以上都不对"],
                "answer_key": "A",
                "rubric": [f"识别出 {keyword}"],
                "rationale": f"材料指出:{keyword}",
                "hint": f"回想材料里关于「{name}」的定义。",
                "source_chunk_ids": chunk_ids[:2],
            }
        return {
            "kind": "short",
            "difficulty": difficulty,
            "stem": f"请用自己的话解释「{name}」,并说明它的作用。",
            "options": [],
            "answer_key": keyword,
            "rubric": [f"提到 {keyword}", f"说明 {name} 的作用"],
            "rationale": f"参考答案要点:{keyword}",
            "hint": f"从「{name}」解决什么问题入手。",
            "source_chunk_ids": chunk_ids[:2],
        }

    @staticmethod
    def _fake_grade(data: dict[str, Any]) -> dict[str, Any]:
        """按 rubric 关键词命中率打分——足以驱动状态机,且完全可预测。"""
        answer = (data.get("answer") or "").lower()
        rubric: list[str] = data.get("rubric") or []
        matched = [r for r in rubric if _loose_hit(r, answer)]
        score = len(matched) / len(rubric) if rubric else (1.0 if answer.strip() else 0.0)
        verdict = "correct" if score >= 0.8 else ("partial" if score >= 0.4 else "incorrect")
        missing = [r for r in rubric if r not in matched]
        return {
            "score": round(score, 3),
            "verdict": verdict,
            "feedback": ("回答覆盖了要点。" if verdict == "correct" else f"还缺:{'、'.join(missing) or '关键依据'}"),
            "matched_rubric": matched,
            "missing_points": missing,
            "misconception_tags": [] if verdict == "correct" else [f"{data.get('concept_name','概念')}-要点缺失"],
        }

    @staticmethod
    def _fake_final_report(data: dict[str, Any]) -> str:
        return (
            f"本次学习覆盖 {data.get('total', 0)} 个概念,"
            f"已掌握 {len(data.get('mastered', []))} 个,"
            f"待巩固 {len(data.get('shaky', []))} 个。"
        )


def _loose_hit(rubric_point: str, answer: str) -> bool:
    """粗糙但确定的关键词匹配:抽取 rubric 里的实词,看答案是否命中。"""
    tokens = [t for t in re.split(r"[\s,,。、;:()()「」【】]+", rubric_point) if len(t) >= 2]
    if not tokens:
        return False
    return any(t.lower() in answer for t in tokens)


def build_llm(force_fake: bool | None = None) -> LLMClient:
    use_fake = RUNTIME.use_fake_llm if force_fake is None else force_fake
    return FakeLLM() if use_fake else AnthropicLLM()


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
