"""Per-episode audit artefacts proving plan fidelity.

Each round directory receives:

    state_input.json         -- the exact battle state handed to the model
    glm_raw_response.json    -- raw model output (+ latency / call count)
    accepted_plan.json       -- the plan exactly as accepted (raw payload)
    plan_sha256.txt          -- SHA256 of the accepted plan file content
    validation_report.json   -- validator PASS/REJECT and error list
    execution_trace.jsonl    -- one line per planned command (executed or not)
    llm_metrics.json         -- call counts, sizes, execution statistics

The accepted plan file is written once and never rewritten; the executor only
appends trace lines.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


class AuditWriter:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._round_dir: Path | None = None
        self._round_index = 0
        self._trace_path: Path | None = None
        self._accepted_plan_bytes: bytes | None = None

    # ------------------------------------------------------------------

    def new_round(self) -> Path:
        self._round_index += 1
        self._round_dir = self.root / f"round_{self._round_index:03d}"
        self._round_dir.mkdir(parents=True, exist_ok=False)
        self._trace_path = self._round_dir / "execution_trace.jsonl"
        self._accepted_plan_bytes = None
        return self._round_dir

    @property
    def round_dir(self) -> Path:
        if self._round_dir is None:
            raise RuntimeError("new_round() must be called first")
        return self._round_dir

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)

    def _write_json(self, name: str, value: Mapping[str, Any] | list[Any]) -> None:
        text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        self._atomic_text(self.round_dir / name, text)

    # ------------------------------------------------------------------

    def write_state_input(self, state: Mapping[str, Any]) -> None:
        self._write_json("state_input.json", dict(state))

    def write_raw_response(
        self,
        *,
        content: str,
        mock: bool,
        latency_s: float,
        prompt_chars: int,
    ) -> None:
        self._write_json(
            "glm_raw_response.json",
            {
                "mock": bool(mock),
                "call_count": 1,
                "latency_s": float(latency_s),
                "prompt_chars": int(prompt_chars),
                "response_chars": len(content),
                "content": content,
                "created_at": datetime.now().astimezone().isoformat(),
            },
        )

    def write_validation_report(self, report: Mapping[str, Any]) -> None:
        self._write_json("validation_report.json", dict(report))

    def write_accepted_plan(self, raw_payload: Mapping[str, Any]) -> str:
        """Write the plan exactly as accepted and return its SHA256."""

        if self._accepted_plan_bytes is not None:
            raise RuntimeError("accepted plan was already written for this round")
        text = json.dumps(raw_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        data = text.encode("utf-8")
        self._accepted_plan_bytes = data
        self._atomic_text(self.round_dir / "accepted_plan.json", text)
        digest = hashlib.sha256(data).hexdigest()
        self._atomic_text(self.round_dir / "plan_sha256.txt", digest + "\n")
        return digest

    @property
    def accepted_plan_sha256(self) -> str | None:
        if self._accepted_plan_bytes is None:
            return None
        return hashlib.sha256(self._accepted_plan_bytes).hexdigest()

    # ------------------------------------------------------------------

    def append_trace(self, record: Mapping[str, Any]) -> None:
        if self._trace_path is None:
            raise RuntimeError("new_round() must be called before tracing")
        line = json.dumps(dict(record), ensure_ascii=False, allow_nan=False)
        with self._trace_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def write_metrics(self, metrics: Mapping[str, Any]) -> None:
        self._write_json("llm_metrics.json", dict(metrics))
