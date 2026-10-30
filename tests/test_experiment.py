import json

import pytest

torch = pytest.importorskip("torch")

from tuneproof.bundle import verify  # noqa: E402
from tuneproof.cli import main  # noqa: E402
from tuneproof.experiment import ExperimentConfig, evaluate, run_experiment, task_data  # noqa: E402
from tuneproof.model import ModelConfig, TinyLM, parameter_hash  # noqa: E402
from tuneproof.report import render_report  # noqa: E402

pytestmark = [pytest.mark.ml, pytest.mark.integration]


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    output = tmp_path_factory.mktemp("run") / "experiment"
    config = ExperimentConfig(pretrain_steps=3, adapter_steps=3, threads=1)
    model_config = ModelConfig(width=16, heads=2, layers=1, rank=2)
    return output, run_experiment(output, config, model_config)


def test_real_experiment_integrity_and_files(measured):
    output, results = measured
    assert all(results["checks"].values())
    assert results["max_control_logit_delta"] == 0
    assert verify(output)["ok"]
    assert results["zero_lr"]["response_nll"] == results["base"]["response_nll"]
    assert results["base"]["examples"] == 32


def test_checkpoint_reload_reproduces_metrics(measured):
    output, results = measured
    model = TinyLM(ModelConfig(**results["model"]))
    model.load_state_dict(torch.load(output / "adapted.pt", weights_only=True))
    model.mode(True)
    assert parameter_hash(model, adapter=True) == results["adapter_parameter_sha256"]
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert evaluate(model, task_data()[1]) == results["lora"]
    finally:
        torch.set_num_threads(old_threads)


def test_seed_reproducibility_and_rng_restoration(measured, tmp_path):
    _, first = measured
    torch.manual_seed(777)
    before = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    second = run_experiment(
        tmp_path / "again", ExperimentConfig(**first["config"]), ModelConfig(**first["model"])
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    assert first["base"] == second["base"] and first["lora"] == second["lora"]
    assert first["adapter_parameter_sha256"] == second["adapter_parameter_sha256"]


def test_report_escapes_untrusted_text(measured):
    output, results = measured
    copied = json.loads(json.dumps(results))
    copied["lora"]["samples"][0]["prompt"] = '<script>alert("bad")</script>'
    report = render_report(copied, json.loads((output / "curves.json").read_text()))
    assert '<script>alert("bad")</script>' not in report
    assert "&lt;script&gt;" in report
    assert "data-toggle" in report and "filter" in report


def test_report_rejects_invented_nonfinite_curve(measured):
    with pytest.raises(ValueError, match="finite"):
        render_report(measured[1], [{"phase": "lora", "step": 1, "loss": float("nan")}])


def test_experiment_cli_smoke(tmp_path, capsys):
    output = tmp_path / "cli"
    assert (
        main(
            [
                "experiment",
                "--output",
                str(output),
                "--pretrain-steps",
                "1",
                "--adapter-steps",
                "1",
                "--threads",
                "1",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert all(result["checks"].values()) and (output / "report.html").exists()
    assert main(["experiment", "--output", str(output), "--adapter-steps", "0"]) == 1
