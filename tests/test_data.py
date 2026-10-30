import json

import pytest

from tuneproof.data import Row, audit, canonical, components, normalize, split


def write(tmp_path, records):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    return path


def test_alpaca_keeps_context():
    row = normalize({"instruction": "Explain", "input": "Python", "output": "Language"}, 7)
    assert row.prompt == "Explain\nInput: Python"
    assert row.line == 7
    assert row.alpaca()["output"] == "Language"


@pytest.mark.parametrize(
    "messages,prompt",
    [
        ([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}], "Q"),
        (
            [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
            "System: S\nUser: Q",
        ),
    ],
)
def test_chatml(messages, prompt):
    row = normalize({"messages": messages}, 1)
    assert row.prompt == prompt and row.response == "A"


@pytest.mark.parametrize(
    "record",
    [
        [],
        None,
        1,
        {},
        {"instruction": "Q", "output": 42},
        {"instruction": " ", "output": "A"},
        {"instruction": "Q", "output": ""},
        {"instruction": "Q", "output": "A", "input": []},
        {"messages": {}},
        {"messages": [1]},
        {"messages": []},
        {"messages": [{"role": "assistant", "content": "A"}]},
        {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "A"}]},
        {"instruction": "Q", "output": "\ud800"},
    ],
)
def test_invalid_schema(record):
    with pytest.raises(ValueError):
        normalize(record, 1)


def test_unicode_canonicalization():
    assert canonical("  ＰＹＴＨＯＮ\tCafe\u0301 ") == canonical("python café")
    assert Row(1, "ҚАЛАЙ СЕН", "x").prompt_hash == Row(2, "қалай  сен", "y").prompt_hash


def test_malformed_line_report_does_not_echo_content(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('\n{"secret-value"\n[]\n', encoding="utf-8")
    result = audit(path)
    assert result.errors[0] == {"line": 2, "reason": "invalid JSON"}
    assert len(result.errors) == 2
    assert "secret-value" not in json.dumps(result.summary())


def test_bom_and_blank_lines(tmp_path):
    path = tmp_path / "bom.jsonl"
    path.write_text('\ufeff\n{"instruction":"Q","output":"A"}\n\n', encoding="utf-8")
    result = audit(path)
    assert len(result.rows) == 1 and result.rows[0].line == 2


def test_conflicting_duplicates_block_split(tmp_path):
    path = write(
        tmp_path,
        [
            {"instruction": "SAME", "output": "A"},
            {"instruction": " same ", "output": "B"},
            {"instruction": "Other", "output": "C"},
        ],
    )
    result = audit(path)
    assert result.conflicts == [[1, 2]]
    with pytest.raises(ValueError, match="conflicting"):
        split(result)


def test_components_are_transitive():
    # A-B and B-C exceed 0.45; A-C do not. All three must stay together.
    rows = [Row(1, "a b c d", "x"), Row(2, "a b c d e", "x"), Row(3, "b c d e f", "x")]
    assert components(rows, 0.45) == [[0, 1, 2]]


def test_group_safe_reproducible_split(tmp_path):
    records = [
        {"instruction": p, "output": "yes"}
        for p in [
            "same prompt",
            "SAME PROMPT",
            "red apple",
            "blue ocean",
            "green grass",
            "cold ice",
        ]
    ]
    result = audit(write(tmp_path, records))
    train, val = split(result, 0.3, 12)
    assert (train, val) == split(result, 0.3, 12)
    assert train and val and len(train) + len(val) == 6
    train_ids = {r.line for r in train}
    for group in result.groups:
        ids = {result.rows[i].line for i in group}
        assert not ids & train_ids or ids <= train_ids


@pytest.mark.parametrize("threshold", [0, -1, 1.1, float("nan"), float("inf")])
def test_bad_threshold(threshold):
    with pytest.raises(ValueError):
        components([], threshold)


@pytest.mark.parametrize("fraction", [0, 1, -1, float("nan")])
def test_bad_fraction(tmp_path, fraction):
    with pytest.raises(ValueError):
        split(audit(write(tmp_path, [])), fraction)


def test_impossible_or_empty_split(tmp_path):
    for records in [[], [{"instruction": "Q", "output": "A"}]]:
        with pytest.raises(ValueError, match="independent"):
            split(audit(write(tmp_path, records)))


def test_invalid_rows_cannot_be_silently_dropped(tmp_path):
    path = write(tmp_path, [{"instruction": "Q", "output": "A"}, []])
    with pytest.raises(ValueError, match="invalid"):
        split(audit(path))


def test_limits_and_bad_encoding(tmp_path):
    path = tmp_path / "large.jsonl"
    path.write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="MiB"):
        audit(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        audit(path)
    path.write_text("{}\n" * 2001)
    with pytest.raises(ValueError, match="2000"):
        audit(path)
