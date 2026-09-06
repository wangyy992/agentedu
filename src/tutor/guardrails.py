"""公网部署用的护栏。

一个会调用付费模型的公开端点,如果不设上限,等于把钱包挂在网上。
这里管四件事:

1. **路径读取** —— HTTP 接口绝不接受任意服务器路径(未授权任意文件读取),
   只接受粘贴文本和内置示例名;本地模式下可以放行一个白名单根目录。
2. **材料长度** —— 上传的材料直接决定每次调用的 token 数,必须有硬上限。
3. **限流** —— 按 IP 的滑动窗口,挡住脚本刷。
4. **每日预算** —— 全局模型调用上限;超了就**降级到离线模型**继续服务,
   而不是返回 500。演示站宁可效果变差,也不要看起来挂了。

进程内存实现,单实例足够。要多实例就把 _WINDOWS / _budget 换成 Redis。
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Limits:
    """全部可用环境变量覆盖,方便按部署环境调紧或调松。"""

    # 演示模式:公网部署时开启,禁用一切按路径读文件的入口
    demo_mode: bool = field(default_factory=lambda: _env_flag("TUTOR_DEMO_MODE"))
    # 允许按路径读取材料的根目录(仅非演示模式生效;留空表示不限制)
    material_root: str = field(default_factory=lambda: os.getenv("TUTOR_MATERIAL_ROOT", ""))

    max_material_chars: int = field(
        default_factory=lambda: _env_int("TUTOR_MAX_MATERIAL_CHARS", 20000))
    min_material_chars: int = field(
        default_factory=lambda: _env_int("TUTOR_MIN_MATERIAL_CHARS", 200))

    # 一次教学循环 = /next + /answer 两个请求,人类手速远到不了 60/分钟,
    # 但足以挡住脚本刷。真要更紧就用环境变量调。
    requests_per_minute: int = field(
        default_factory=lambda: _env_int("TUTOR_RATE_LIMIT_PER_MIN", 60))
    ingests_per_hour: int = field(
        default_factory=lambda: _env_int("TUTOR_INGESTS_PER_HOUR", 5))

    # 每日模型调用总预算。<=0 表示不限制(本地开发默认不限)
    daily_llm_calls: int = field(
        default_factory=lambda: _env_int("TUTOR_DAILY_LLM_CALL_BUDGET", 0))


LIMITS = Limits()


class RateLimitExceeded(Exception):
    def __init__(self, message: str, retry_after: int = 60) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class MaterialRejected(Exception):
    pass


# --- 限流 ---------------------------------------------------------------
class SlidingWindow:
    """按 key 的滑动窗口计数。线程安全(uvicorn 的同步路由跑在线程池里)。"""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                retry = int(self.window - (now - hits[0])) + 1
                raise RateLimitExceeded(
                    f"请求过于频繁({self.window}秒内最多 {self.limit} 次),"
                    f"请 {retry} 秒后再试。", retry_after=retry)
            hits.append(now)
            self._prune(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def _prune(self, now: float) -> None:
        """顺手清掉早已过期的 key,避免长期运行内存无限增长。"""
        if len(self._hits) < 1024:
            return
        stale = [k for k, v in self._hits.items() if not v or now - v[-1] > self.window]
        for k in stale:
            self._hits.pop(k, None)


# --- 每日预算 -----------------------------------------------------------
class DailyBudget:
    """全局模型调用计数,按自然日重置。超额不报错,交由调用方降级。"""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._day = date.today()
        self._used = 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        today = date.today()
        if today != self._day:
            self._day, self._used = today, 0

    def exhausted(self) -> bool:
        if self.limit <= 0:
            return False
        with self._lock:
            self._roll()
            return self._used >= self.limit

    def record(self, calls: int) -> None:
        if self.limit <= 0 or calls <= 0:
            return
        with self._lock:
            self._roll()
            self._used += calls

    def reset(self, limit: int | None = None) -> None:
        with self._lock:
            if limit is not None:
                self.limit = limit
            self._day, self._used = date.today(), 0

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            self._roll()
            return {"limit": self.limit, "used": self._used,
                    "exhausted": self.limit > 0 and self._used >= self.limit}


# --- 材料校验 -----------------------------------------------------------
def validate_material_text(text: str, limits: Limits | None = None) -> str:
    # 注意:不能写成 `limits: Limits = LIMITS`——默认值在 import 时就求值,
    # 之后再改模块级 LIMITS 会静默不生效。这里在调用时才读。
    limits = limits or LIMITS
    text = (text or "").strip()
    if len(text) < limits.min_material_chars:
        raise MaterialRejected(
            f"材料太短(至少 {limits.min_material_chars} 字),抽不出有意义的知识点。")
    if len(text) > limits.max_material_chars:
        raise MaterialRejected(
            f"材料太长({len(text)} 字,上限 {limits.max_material_chars} 字)。"
            "演示站对长度有限制,本地运行不受此限。")
    return text


def resolve_material_path(raw: str, limits: Limits | None = None) -> Path:
    """把用户给的 path 解析成一个**允许读取**的真实路径。

    演示模式下直接拒绝:HTTP 接口接受任意服务器路径 = 未授权任意文件读取。
    非演示模式下,如果配了 TUTOR_MATERIAL_ROOT,则必须落在该目录内
    (先 resolve 再比较,挡住 ../ 穿越和符号链接)。
    """
    limits = limits or LIMITS
    if limits.demo_mode:
        raise MaterialRejected(
            "演示站不支持按服务器路径读取材料。请直接粘贴文本,或选用内置示例。")

    path = Path(raw).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MaterialRejected(f"找不到材料文件:{raw}") from exc
    if not resolved.is_file():
        raise MaterialRejected(f"不是一个文件:{raw}")

    if limits.material_root:
        root = Path(limits.material_root).expanduser().resolve()
        if not resolved.is_relative_to(root):
            raise MaterialRejected(f"只允许读取 {root} 目录下的材料。")
    return resolved


# --- 全局实例 -----------------------------------------------------------
request_limiter = SlidingWindow(LIMITS.requests_per_minute, 60)
ingest_limiter = SlidingWindow(LIMITS.ingests_per_hour, 3600)
budget = DailyBudget(LIMITS.daily_llm_calls)


def reset_all(requests_per_minute: int | None = None,
              ingests_per_hour: int | None = None,
              daily_llm_calls: int | None = None) -> None:
    """清空所有计数器,可选地改上限。测试用——生产不会调它。"""
    request_limiter.reset()
    ingest_limiter.reset()
    if requests_per_minute is not None:
        request_limiter.limit = requests_per_minute
    if ingests_per_hour is not None:
        ingest_limiter.limit = ingests_per_hour
    budget.reset(daily_llm_calls)


def client_key(request) -> str:
    """取调用方标识。部署在反代后面时优先用 X-Forwarded-For 的第一跳。"""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return getattr(getattr(request, "client", None), "host", "") or "unknown"
