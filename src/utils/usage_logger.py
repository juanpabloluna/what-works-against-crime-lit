"""Usage logger — records who uses the system and what they query."""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

LOG_PATH = Path(__file__).parent.parent.parent / "data" / "usage_log.jsonl"

_lock = threading.Lock()


def log_usage(
    user: str,
    page: str,
    query: str,
    extra: Optional[dict] = None,
) -> None:
    """Append a usage entry to the JSONL log file.

    Args:
        user: Name/identifier of the user.
        page: Which page was used (qa, synthesis, review, agentic_qa).
        query: The query or topic submitted.
        extra: Optional dict with additional info (e.g. filters used).
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user": user,
        "page": page,
        "query": query,
    }
    if extra:
        entry["extra"] = extra

    try:
        with _lock:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write usage log: {e}")


def read_usage_log() -> list[dict]:
    """Read all usage log entries."""
    if not LOG_PATH.exists():
        return []
    entries = []
    try:
        with open(LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Failed to read usage log: {e}")
    return entries
