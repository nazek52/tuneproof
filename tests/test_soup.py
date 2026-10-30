"""Optional contract test against the real, pinned Soup schema. No GPU/model."""

import json

import pytest
import yaml

from tuneproof.bundle import prepare


def test_export_validates_against_soup_0771_schema(tmp_path):
    schema = pytest.importorskip("soup_cli.config.schema")
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"instruction": p, "output": "Answer"})
            for p in ["red apple", "blue sky", "green grass"]
        )
    )
    output = tmp_path / "bundle"
    prepare(source, output, base="HuggingFaceTB/SmolLM2-135M-Instruct")
    config = schema.SoupConfig.model_validate(yaml.safe_load((output / "soup.yaml").read_text()))
    assert config.data.val_split == 0
    assert config.training.lora.r == 8
