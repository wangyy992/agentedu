"""全局配置:模型、参数、路径。

模型 ID 集中在这里,避免散落各处。默认使用 claude-opus-5。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# --- .env 加载 ----------------------------------------------------------
def _load_dotenv() -> None:
    """从项目根目录的 .env 读取配置(如 ANTHROPIC_API_KEY)。

    自己实现而不是引入 python-dotenv:只有十几行,省一个依赖,
    也和这个项目「能不加依赖就不加」的取向一致(检索也是同样的理由用了 BM25)。

    真实环境变量优先于 .env——这样 Docker / CI 里传进来的值不会被文件覆盖。
    必须在本模块读取任何环境变量之前调用。
    """
    for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
        env_file = base / ".env"
        if not env_file.is_file():
            continue
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
        return


_load_dotenv()


# --- 模型 ---------------------------------------------------------------
MODEL = os.getenv("TUTOR_MODEL", "claude-opus-5")

# 每类任务的 effort 不同:规划/答疑需要更强推理,出题/判分可以省一点。
# effort 走 output_config.effort(low | medium | high | xhigh | max)
EFFORT_BY_TASK: dict[str, str] = {
    "extract_concepts": "high",
    "explain": "high",
    "write_item": "medium",
    "grade": "medium",
    "answer_question": "high",
    "final_report": "medium",
}

MAX_TOKENS = 16000


# --- 教学策略参数 -------------------------------------------------------
@dataclass(frozen=True)
class TutorParams:
    """自适应策略的可调参数。集中放这里,方便做消融实验。"""

    # 贝叶斯知识追踪(BKT)
    p_init: float = 0.15   # 先验掌握概率
    p_transit: float = 0.25  # 一次有效练习后的学会概率
    p_slip: float = 0.10   # 会了但答错(手滑)
    p_guess: float = 0.20  # 不会但蒙对

    # 推进 / 补救阈值
    mastery_threshold: float = 0.85   # BKT 掌握概率阈值
    min_attempts_for_mastery: int = 3  # 达标至少要做够的题量(防止蒙对两道就放行)
    min_difficulty_for_mastery: int = 3  # 达标至少要做对过的难度(应用层级,不能只考记忆)
    remediate_threshold: float = 0.45  # 低于此分触发针对性补救
    max_items_per_concept: int = 6     # 防止在一个概念上死循环

    # 难度(1~5,对应 Bloom:记忆/理解/应用/分析/综合)
    min_difficulty: int = 1
    max_difficulty: int = 5

    # 间隔复习:掌握后隔多少步回头抽查
    review_delay_steps: int = 4
    review_probability_floor: float = 0.7  # 复习题答错会把掌握度打回这个比例
    max_reviews_per_concept: int = 2       # 复习次数上限,防止在复习队列里来回打转

    # 检索
    retrieve_top_k: int = 5


PARAMS = TutorParams()


# --- 路径 ---------------------------------------------------------------
def _data_dir() -> Path:
    d = Path(os.getenv("TUTOR_DATA_DIR", Path.cwd() / ".tutor_data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


DATA_DIR = _data_dir()
DB_PATH = DATA_DIR / "tutor.db"


@dataclass
class RuntimeConfig:
    """运行期开关。use_fake_llm=True 时完全离线运行(无需 API Key)。"""

    use_fake_llm: bool = field(
        default_factory=lambda: os.getenv("TUTOR_FAKE_LLM", "").lower() in ("1", "true", "yes")
    )
    verbose: bool = field(
        default_factory=lambda: os.getenv("TUTOR_VERBOSE", "").lower() in ("1", "true", "yes")
    )


RUNTIME = RuntimeConfig()
