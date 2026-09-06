"""随包分发的示例材料。

放在包内而不是仓库根目录,是为了 `pip install .`(Docker 里的标准做法)之后
仍然找得到——根目录的 examples/ 不会被打包进 wheel。
"""
from __future__ import annotations

from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent


def list_examples() -> list[str]:
    return sorted(p.stem for p in EXAMPLES_DIR.glob("*.md"))


def example_path(name: str) -> Path:
    """按名字取内置示例。名字里不允许出现路径分隔符。"""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise KeyError(f"非法的示例名:{name!r}")
    path = EXAMPLES_DIR / f"{Path(name).stem}.md"
    if not path.is_file():
        raise KeyError(f"没有内置示例 {name!r},可用的有:{', '.join(list_examples())}")
    return path


def resolve(path_or_name: str) -> Path:
    """先当普通路径找;找不到再当内置示例名找。

    这样 README 里写 `--example gradient_descent` 和直接给一个本地文件路径
    都能工作,不管是 clone 下来跑还是 pip 装完跑。
    """
    candidate = Path(path_or_name)
    if candidate.is_file():
        return candidate
    return example_path(candidate.stem)
