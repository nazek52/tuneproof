"""Atomic local bundles with explicit SHA-256 evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml

from .data import audit, split


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def fresh_directory(destination: Path):
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".tuneproof-", dir=destination.parent))
    try:
        yield staging
        # Refuse overwriting another writer's destination, even if it appeared during work.
        if destination.exists() or destination.is_symlink():
            raise ValueError("destination appeared during operation")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def manifest(directory: Path, metadata: dict) -> dict:
    result = {
        "schema_version": 1,
        "metadata": metadata,
        "files": {
            p.relative_to(directory).as_posix(): sha256(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
    }
    write_json(directory / "manifest.json", result)
    return result


def verify(directory: Path) -> dict:
    path = directory / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("manifest must be a regular file")
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("manifest exceeds 1 MiB")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported manifest")
    entries = data.get("files")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("manifest has no files")
    failures = []
    for name, expected in entries.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or not relative.parts
            or name == "manifest.json"
        ):
            raise ValueError("unsafe manifest path")
        target = directory / relative
        if any(p.is_symlink() for p in [target, *target.parents] if p != directory.parent):
            raise ValueError("symlink in artifact path")
        if not target.is_file() or sha256(target) != expected:
            failures.append(name)
    actual = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p != path
    }
    unexpected = sorted(actual - entries.keys())
    return {
        "ok": not failures and not unexpected,
        "checked_files": len(entries),
        "failed_files": failures,
        "unexpected_files": unexpected,
    }


def prepare(
    source: Path,
    output: Path,
    *,
    base: str,
    threshold: float = 0.8,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    if not base.strip():
        raise ValueError("base must be nonempty")
    audited = audit(source, threshold)
    train, val = split(audited, val_fraction, seed)
    with fresh_directory(output) as stage:
        for name, rows in [("train.jsonl", train), ("validation.jsonl", val)]:
            (stage / name).write_text(
                "".join(json.dumps(r.alpaca(), ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
            )
        # Soup has no data.eval path in its current schema. Keep the holdout
        # separate and disable its automatic row-level split.
        config = {
            "base": base,
            "task": "sft",
            "data": {
                "train": "./train.jsonl",
                "format": "alpaca",
                "val_split": 0.0,
                "max_length": 256,
            },
            "training": {"epochs": 1, "lr": 0.0002, "batch_size": 1, "lora": {"r": 8, "alpha": 16}},
            "output": "./model-output",
        }
        (stage / "soup.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        summary = audited.summary() | {
            "train_rows": len(train),
            "validation_rows": len(val),
            "seed": seed,
            "requested_val_fraction": val_fraction,
            "actual_val_fraction": len(val) / len(audited.rows),
        }
        write_json(stage / "audit.json", summary)
        manifest(stage, {"kind": "soup-data-bundle", "source_sha256": audited.source_sha256})
    return summary
