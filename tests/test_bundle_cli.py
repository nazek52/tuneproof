import json
import subprocess
import sys

import pytest
import yaml

from tuneproof.bundle import fresh_directory, manifest, prepare, verify, write_json
from tuneproof.cli import main


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "input.jsonl"
    rows = [
        {"instruction": p, "output": "answer"}
        for p in ["red apple", "blue sea", "green grass", "orange fruit", "cold snow"]
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_prepare_complete_bundle(source, tmp_path):
    before = source.read_bytes()
    output = tmp_path / "bundle"
    summary = prepare(source, output, base="test/model")
    assert summary["train_rows"] + summary["validation_rows"] == 5
    assert source.read_bytes() == before
    assert verify(output)["ok"]
    config = yaml.safe_load((output / "soup.yaml").read_text())
    assert config["data"]["val_split"] == 0
    assert "eval" not in config["data"]
    assert config["data"]["format"] == "alpaca"
    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert all(set(row) == {"instruction", "input", "output"} for row in train)


def test_collision_keeps_existing_output(source, tmp_path):
    output = tmp_path / "bundle"
    prepare(source, output, base="test/model")
    original = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        prepare(source, output, base="test/model")
    assert (output / "manifest.json").read_bytes() == original


def test_failure_cleans_staging(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(RuntimeError), fresh_directory(output) as stage:
        (stage / "partial.txt").write_text("partial")
        raise RuntimeError("test error")
    assert not output.exists()
    assert not list(tmp_path.glob(".tuneproof-*"))


def test_new_destination_during_operation(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="appeared"), fresh_directory(output):
        output.mkdir()
        (output / "sentinel").write_text("safe")
    assert (output / "sentinel").read_text() == "safe"


def test_tamper_missing_and_extra_files(source, tmp_path):
    output = tmp_path / "bundle"
    prepare(source, output, base="test/model")
    (output / "train.jsonl").write_text("changed")
    (output / "validation.jsonl").unlink()
    (output / "unexpected.txt").write_text("extra")
    result = verify(output)
    assert not result["ok"]
    assert set(result["failed_files"]) == {"train.jsonl", "validation.jsonl"}
    assert result["unexpected_files"] == ["unexpected.txt"]


@pytest.mark.parametrize("name", ["../secret", "/etc/passwd", "a\\b", "manifest.json"])
def test_manifest_traversal_rejected(tmp_path, name):
    write_json(tmp_path / "manifest.json", {"schema_version": 1, "files": {name: "x"}})
    with pytest.raises(ValueError, match="unsafe"):
        verify(tmp_path)


def test_symlink_refused(tmp_path):
    target = tmp_path / "real"
    target.write_text("x")
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / "linked").symlink_to(target)
    write_json(directory / "manifest.json", {"schema_version": 1, "files": {"linked": "x"}})
    with pytest.raises(ValueError, match="symlink"):
        verify(directory)


@pytest.mark.parametrize(
    "payload", [{}, [], {"schema_version": 1, "files": {}}, {"schema_version": 3}]
)
def test_invalid_manifest(tmp_path, payload):
    write_json(tmp_path / "manifest.json", payload)
    with pytest.raises(ValueError):
        verify(tmp_path)


def test_manifest_roundtrip(tmp_path):
    (tmp_path / "file.txt").write_text("hello")
    manifest(tmp_path, {"test": True})
    assert verify(tmp_path) == {
        "ok": True,
        "checked_files": 1,
        "failed_files": [],
        "unexpected_files": [],
    }


def test_cli_flow_and_errors(source, tmp_path, capsys):
    assert main(["audit", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["valid_rows"] == 5
    output = tmp_path / "bundle"
    assert main(["prepare", str(source), "--output", str(output)]) == 0
    capsys.readouterr()
    assert main(["verify", str(output)]) == 0
    capsys.readouterr()
    (output / "train.jsonl").write_text("changed")
    assert main(["verify", str(output)]) == 2
    capsys.readouterr()
    assert main(["audit", str(tmp_path / "missing")]) == 1
    assert "error" in json.loads(capsys.readouterr().err)


def test_empty_audit_exit_code(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.touch()
    assert main(["audit", str(empty)]) == 2


def test_module_help_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "tuneproof", "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0 and "experiment" in result.stdout


def test_empty_base_is_refused(source, tmp_path):
    with pytest.raises(ValueError, match="base"):
        prepare(source, tmp_path / "out", base=" ")
