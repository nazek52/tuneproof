"""Bounded JSONL audit and lexical, component-aware splitting. No ML dependency."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 2000


def canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Row:
    line: int
    prompt: str
    response: str

    @property
    def prompt_hash(self) -> str:
        return digest(canonical(self.prompt))

    def alpaca(self) -> dict:
        return {"instruction": self.prompt, "input": "", "output": self.response}


@dataclass
class Audit:
    rows: list[Row]
    errors: list[dict]
    groups: list[list[int]]
    conflicts: list[list[int]]
    source_sha256: str
    threshold: float

    def summary(self) -> dict:
        return {
            "schema_version": 1,
            "source_sha256": self.source_sha256,
            "valid_rows": len(self.rows),
            "invalid_rows": len(self.errors),
            "errors": self.errors,
            "prompt_components": len(self.groups),
            "duplicate_or_similar_rows": sum(len(g) - 1 for g in self.groups),
            "conflicting_prompt_lines": self.conflicts,
            "threshold": self.threshold,
            "method": "NFKC casefold prompt equality + word-bigram Jaccard components",
            "limitations": "Lexical only; no semantic or privacy-safety guarantee.",
            "components": [[self.rows[i].line for i in group] for group in self.groups],
        }


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    # Lone surrogates are legal JSON escapes but not valid UTF-8 training text.
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be valid UTF-8 text") from exc
    return value.strip()


def normalize(record: object, line: int) -> Row:
    if not isinstance(record, dict):
        raise ValueError("row must be a JSON object")
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise ValueError("messages must be a list of objects")
        roles = [m.get("role") for m in messages]
        if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
            raise ValueError("only single-turn ChatML with optional system message is supported")
        texts = [_text(m.get("content"), "message content") for m in messages]
        prompt = texts[-2] if len(texts) == 2 else f"System: {texts[0]}\nUser: {texts[1]}"
        return Row(line, prompt, texts[-1])
    instruction = _text(record.get("instruction"), "instruction")
    context = record.get("input", "")
    if not isinstance(context, str):
        raise ValueError("input must be a string")
    if context.strip():
        context = _text(context, "input")
        instruction += f"\nInput: {context}"
    return Row(line, instruction, _text(record.get("output"), "output"))


def shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", canonical(text))
    return set(zip(words, words[1:], strict=False)) if len(words) > 1 else {(text,)}


def components(rows: list[Row], threshold: float = 0.8) -> list[list[int]]:
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be finite and in (0, 1]")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"at most {MAX_ROWS} rows are supported")
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [r.prompt_hash for r in rows]
    sets = [shingles(r.prompt) for r in rows]
    for i in range(len(rows)):
        for j in range(i):
            union = sets[i] | sets[j]
            similarity = len(sets[i] & sets[j]) / len(union) if union else 0
            if hashes[i] == hashes[j] or similarity >= threshold:
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(rows)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def audit(path: Path, threshold: float = 0.8) -> Audit:
    # Read at most cap+1 bytes, even if the source grows after stat().
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("input exceeds 10 MiB")
    try:
        lines = raw.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("input must be UTF-8") from exc
    rows, errors = [], []
    nonblank = 0
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        nonblank += 1
        if nonblank > MAX_ROWS:
            raise ValueError(f"at most {MAX_ROWS} nonblank lines are supported")
        try:
            rows.append(normalize(json.loads(line), number))
        except (ValueError, RecursionError) as exc:
            # Fixed parser error avoids echoing user data into the audit.
            reason = "invalid JSON" if isinstance(exc, json.JSONDecodeError) else str(exc)
            errors.append({"line": number, "reason": reason})
    by_prompt: dict[str, list[Row]] = {}
    for row in rows:
        by_prompt.setdefault(row.prompt_hash, []).append(row)
    conflicts = [
        [r.line for r in group]
        for group in by_prompt.values()
        if len({canonical(r.response) for r in group}) > 1
    ]
    return Audit(
        rows,
        errors,
        components(rows, threshold),
        conflicts,
        hashlib.sha256(raw).hexdigest(),
        threshold,
    )


def split(audited: Audit, val_fraction: float = 0.2, seed: int = 42) -> tuple[list[Row], list[Row]]:
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be finite and between 0 and 1")
    if audited.errors or audited.conflicts:
        raise ValueError("fix invalid rows and conflicting exact prompts before preparing data")
    if len(audited.groups) < 2:
        raise ValueError("need at least two independent prompt components for a safe split")
    groups = list(audited.groups)
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(audited.rows) * val_fraction))
    selected: list[int] = []
    for group in groups[:-1]:
        selected.extend(group)
        if len(selected) >= target:
            break
    val_indices = set(selected)
    train = [row for i, row in enumerate(audited.rows) if i not in val_indices]
    val = [row for i, row in enumerate(audited.rows) if i in val_indices]
    return train, val


def row_dict(row: Row) -> dict:
    return asdict(row)
