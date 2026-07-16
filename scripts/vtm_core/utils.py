from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def safe_name(text: str, fallback: str = "video") -> str:
    text = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return (text[:100].rstrip() or fallback)


def topic_slug(text: str, fallback: str = "frame") -> str:
    words = re.findall(r"[\w\u3400-\u9fff]+", text, flags=re.UNICODE)
    return "-".join(words[:6])[:48] or fallback


def timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes:02d}m{secs:02d}s"


def require_command(name: str) -> str:
    result = shutil.which(name)
    if not result:
        raise RuntimeError(f"Required command not found: {name}")
    return result
