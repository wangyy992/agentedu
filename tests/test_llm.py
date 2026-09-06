"""用桩 client 测**真实的** AnthropicLLM:缓存布局、JSON 重试、tool 循环。

这样这些逻辑不会因为"只在联网时才走到"而失去覆盖。
"""
import json
import types

import pytest

from tutor.llm import AnthropicLLM, Prompt
from tutor.schemas import Grade


def block(kind, **kw):
    return types.SimpleNamespace(type=kind, **kw)


class StubClient:
    """按脚本依次返回预设响应,并记录收到的请求。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # messages 是同一个 list 对象且会被后续轮次继续 append,
        # 这里必须存快照,否则断言看到的是"最终态"而不是"这次发出去的内容"。
        self.requests.append({**kwargs, "messages": list(kwargs.get("messages", []))})
        return self._responses.pop(0)


def text_response(text, stop_reason="end_turn"):
    return types.SimpleNamespace(
        content=[block("text", text=text)],
        stop_reason=stop_reason,
        usage=types.SimpleNamespace(input_tokens=10, output_tokens=5,
                                    cache_read_input_tokens=7, cache_creation_input_tokens=0),
    )


def tool_response(name, args, tool_id="tu_1"):
    return types.SimpleNamespace(
        content=[block("tool_use", id=tool_id, name=name, input=args)],
        stop_reason="tool_use",
        usage=types.SimpleNamespace(input_tokens=10, output_tokens=5,
                                    cache_read_input_tokens=0, cache_creation_input_tokens=3),
    )


GOOD_GRADE = json.dumps({
    "score": 0.5, "verdict": "partial", "feedback": "还行",
    "matched_rubric": [], "missing_points": [], "misconception_tags": [],
})


# --- 缓存布局 -----------------------------------------------------------
def test_material_sits_behind_cache_breakpoint_and_task_prompt_after_it():
    stub = StubClient([text_response(GOOD_GRADE)])
    llm = AnthropicLLM(client=stub)
    llm.structured(Prompt("grade", "任务指令", "用户消息"), Grade, cached_context="材料全文")

    system = stub.requests[0]["system"]
    assert len(system) == 3
    assert "材料全文" in system[1]["text"]
    assert system[1]["cache_control"]["type"] == "ephemeral"
    # 易变的任务指令必须排在缓存块之后,否则前缀每次都变、缓存全废
    assert system[2]["text"] == "任务指令"
    assert "cache_control" not in system[2]


def test_no_cache_block_when_no_material():
    stub = StubClient([text_response(GOOD_GRADE)])
    AnthropicLLM(client=stub).structured(Prompt("grade", "s", "u"), Grade)
    assert len(stub.requests[0]["system"]) == 2


def test_usage_is_accumulated():
    stub = StubClient([text_response(GOOD_GRADE), text_response(GOOD_GRADE)])
    llm = AnthropicLLM(client=stub)
    llm.structured(Prompt("grade", "s", "u"), Grade)
    llm.structured(Prompt("grade", "s", "u"), Grade)
    assert llm.usage.calls == 2
    assert llm.usage.cache_read == 14


# --- 结构化输出与重试 ---------------------------------------------------
def test_schema_is_sent_and_effort_applied():
    stub = StubClient([text_response(GOOD_GRADE)])
    AnthropicLLM(client=stub).structured(Prompt("grade", "s", "u"), Grade)
    cfg = stub.requests[0]["output_config"]
    assert cfg["format"]["type"] == "json_schema"
    assert cfg["format"]["schema"]["additionalProperties"] is False
    assert cfg["effort"] == "medium"          # config.EFFORT_BY_TASK["grade"]
    assert stub.requests[0]["thinking"] == {"type": "adaptive"}


def test_invalid_json_is_repaired_on_second_attempt():
    stub = StubClient([text_response("这不是 JSON"), text_response(GOOD_GRADE)])
    result = AnthropicLLM(client=stub).structured(Prompt("grade", "s", "u"), Grade)
    assert result.score == 0.5
    assert len(stub.requests) == 2
    # 第二次请求把错误回灌给模型
    assert "不符合 schema" in stub.requests[1]["messages"][-1]["content"]


def test_two_bad_responses_raise():
    stub = StubClient([text_response("bad"), text_response("still bad")])
    with pytest.raises(RuntimeError, match="不符合 schema"):
        AnthropicLLM(client=stub).structured(Prompt("grade", "s", "u"), Grade)


# --- tool-use 循环 ------------------------------------------------------
def test_tool_loop_executes_and_feeds_result_back():
    stub = StubClient([
        tool_response("search_material", {"query": "学习率", "top_k": 3}),
        text_response("学习率过大会震荡 [c003]"),
    ])
    calls = []

    def executor(name, args):
        calls.append((name, args))
        return "[c003] 学习率过大会震荡"

    result = AnthropicLLM(client=stub).tool_loop(
        Prompt("answer_question", "s", "u"), tools=[], executor=executor)

    assert calls == [("search_material", {"query": "学习率", "top_k": 3})]
    assert result.text == "学习率过大会震荡 [c003]"
    assert len(result.tool_calls) == 1
    # 工具结果必须以 tool_result 块回传到同一条 user 消息里
    fed = stub.requests[1]["messages"][-1]["content"]
    assert fed[0]["type"] == "tool_result" and fed[0]["tool_use_id"] == "tu_1"


def test_tool_failure_is_reported_back_not_swallowed():
    stub = StubClient([
        tool_response("search_material", {"query": "x", "top_k": 3}),
        text_response("材料里没查到"),
    ])

    def executor(name, args):
        raise RuntimeError("索引挂了")

    AnthropicLLM(client=stub).tool_loop(Prompt("answer_question", "s", "u"),
                                        tools=[], executor=executor)
    fed = stub.requests[1]["messages"][-1]["content"][0]
    assert fed["is_error"] is True and "索引挂了" in fed["content"]


def test_tool_loop_stops_at_max_turns():
    """模型一直要调工具时必须能收敛,并且最后一次不带 tools 强制收口。"""
    stub = StubClient([tool_response("search_material", {"query": "q", "top_k": 1})] * 3
                      + [text_response("最终回答")])
    result = AnthropicLLM(client=stub).tool_loop(
        Prompt("answer_question", "s", "u"), tools=[], executor=lambda n, a: "[c001] x",
        max_turns=3)
    assert result.text == "最终回答"
    assert len(result.tool_calls) == 3
    assert "不要再调用工具" in stub.requests[-1]["messages"][-1]["content"]
