"""Seeded, real CPU fine-tuning with a matched zero-learning-rate control."""

from __future__ import annotations

import copy
import math
import platform
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .bundle import fresh_directory, manifest, write_json
from .data import canonical, digest
from .model import (
    VOCAB,
    ModelConfig,
    TinyLM,
    encode_batch,
    is_adapter,
    parameter_hash,
    response_loss,
)
from .report import render_report


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    pretrain_steps: int = 180
    adapter_steps: int = 240
    batch_size: int = 24
    learning_rate: float = 0.008
    threads: int = 2

    def __post_init__(self):
        if not 1 <= self.pretrain_steps <= 5000 or not 1 <= self.adapter_steps <= 5000:
            raise ValueError("steps must be in [1, 5000]")
        if not 1 <= self.batch_size <= 128 or not 1 <= self.threads <= 16:
            raise ValueError("invalid batch_size or threads")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.1:
            raise ValueError("learning_rate must be finite and in (0, 0.1]")


def task_data() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Synthetic routing. Held-out ticket identifiers never occur in training."""
    intents = {"refund": "BILLING", "password": "ACCOUNT", "crash": "TECH", "delivery": "SHIPPING"}
    train, val = [], []
    for keyword, label in intents.items():
        for i in range(40):
            pair = (f"Ticket {i:03d}: help with {keyword}", label)
            (train if i < 32 else val).append(pair)
    return train, val


def fit(
    model: TinyLM,
    pairs: list[tuple[str, str]],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    phase: str,
) -> list[dict]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    model.train()
    history = []
    for step in range(1, steps + 1):
        batch = [pairs[rng.randrange(len(pairs))] for _ in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        loss = response_loss(model, batch)
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == steps:
            history.append({"phase": phase, "step": step, "loss": float(loss.detach())})
    return history


@torch.no_grad()
def evaluate(model: TinyLM, pairs: list[tuple[str, str]]) -> dict:
    model.eval()
    loss_sum, token_count = 0.0, 0
    for start in range(0, len(pairs), 32):
        x, y = encode_batch(pairs[start : start + 32], model.config.context)
        loss_sum += float(
            F.cross_entropy(
                model(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100, reduction="sum"
            )
        )
        token_count += int((y != -100).sum())
    samples = [{"prompt": p, "expected": r, "generated": model.generate(p)} for p, r in pairs]
    correct = sum(s["generated"] == s["expected"] for s in samples)
    labels = {response for _, response in pairs}
    # Measure routing separately from compact formatting. A verbose but correct
    # base answer should not be presented as new semantic capability after LoRA.
    label_correct = sum(
        set(re.findall(r"[A-Z]+", s["generated"])) & labels == {s["expected"]} for s in samples
    )
    return {
        "response_nll": loss_sum / token_count,
        "response_tokens": token_count,
        "exact_match": correct / len(pairs),
        "correct": correct,
        "routing_label_match": label_correct / len(pairs),
        "examples": len(pairs),
        "samples": samples,
    }


def run_experiment(
    output: Path,
    config: ExperimentConfig | None = None,
    model_config: ModelConfig | None = None,
) -> dict:
    config = config or ExperimentConfig()
    model_config = model_config or ModelConfig()
    started = time.perf_counter()
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        # fork_rng restores caller RNG after the experiment.
        with torch.random.fork_rng(devices=[]), fresh_directory(output) as stage:
            torch.manual_seed(config.seed)
            train, val = task_data()
            overlap = {canonical(p) for p, _ in train} & {canonical(p) for p, _ in val}
            if overlap:
                raise ValueError("synthetic task has exact prompt leakage")
            for name, pairs in [("train", train), ("validation", val)]:
                write_json(stage / f"{name}.json", [{"prompt": p, "response": r} for p, r in pairs])
            base = TinyLM(model_config)
            base.mode(False)
            verbose_train = [(p, f"The route is {r}.") for p, r in train]
            curve = fit(
                base,
                verbose_train,
                steps=config.pretrain_steps,
                batch_size=config.batch_size,
                lr=0.003,
                seed=config.seed,
                phase="pretrain",
            )
            base_metrics = evaluate(base, val)
            base_hash = parameter_hash(base, adapter=False)
            adapted, control = copy.deepcopy(base), copy.deepcopy(base)
            adapted.mode(True)
            control.mode(True)
            adapter_before = parameter_hash(adapted, adapter=True)
            curve += fit(
                adapted,
                train,
                steps=config.adapter_steps,
                batch_size=config.batch_size,
                lr=config.learning_rate,
                seed=config.seed + 1,
                phase="lora",
            )
            curve += fit(
                control,
                train,
                steps=config.adapter_steps,
                batch_size=config.batch_size,
                lr=0.0,
                seed=config.seed + 1,
                phase="zero_lr",
            )
            adapted_metrics, control_metrics = evaluate(adapted, val), evaluate(control, val)
            x, _ = encode_batch(val[:4], model_config.context)
            with torch.no_grad():
                max_logit_delta = float((base(x) - control(x)).abs().max())
            checks = {
                "frozen_base_unchanged": base_hash == parameter_hash(adapted, adapter=False),
                "zero_lr_base_unchanged": base_hash == parameter_hash(control, adapter=False),
                "zero_lr_adapter_unchanged": adapter_before
                == parameter_hash(control, adapter=True),
                "adapter_changed": adapter_before != parameter_hash(adapted, adapter=True),
                "zero_lr_logits_identical": max_logit_delta == 0.0,
                "no_exact_prompt_overlap": not overlap,
            }
            if not all(checks.values()):
                raise ValueError(f"experiment integrity checks failed: {checks}")
            total = sum(p.numel() for p in adapted.parameters())
            trainable = sum(p.numel() for n, p in adapted.named_parameters() if is_adapter(n))
            results = {
                "schema_version": 1,
                "kind": "measured-cpu-experiment",
                "task": "synthetic ticket routing; verbose-to-label format adaptation",
                "config": asdict(config),
                "model": asdict(model_config),
                "environment": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "platform": platform.platform(),
                    "device": "cpu",
                },
                "parameters": {
                    "total": total,
                    "adapter": trainable,
                    "adapter_percent": 100 * trainable / total,
                },
                "data": {
                    "train_rows": len(train),
                    "validation_rows": len(val),
                    "train_prompt_digest": digest("\n".join(p for p, _ in train)),
                    "validation_prompt_digest": digest("\n".join(p for p, _ in val)),
                    "split_policy": (
                        "ticket IDs 000-031 train, 032-039 validation; shared templates"
                    ),
                },
                "base": base_metrics,
                "lora": adapted_metrics,
                "zero_lr": control_metrics,
                "checks": checks,
                "max_control_logit_delta": max_logit_delta,
                "base_parameter_sha256": base_hash,
                "adapter_parameter_sha256": parameter_hash(adapted, adapter=True),
                "duration_seconds": time.perf_counter() - started,
                "limitations": [
                    "Tiny model on synthetic data, not a pretrained LLM benchmark.",
                    "Shared lexical templates make this an easy in-distribution task.",
                    "One seed; no statistical significance or general capability claim.",
                    "Custom checkpoints are not PEFT/Soup-compatible adapters.",
                ],
            }
            write_json(stage / "results.json", results)
            write_json(stage / "curves.json", curve)
            torch.save(base.state_dict(), stage / "base.pt")
            torch.save(adapted.state_dict(), stage / "adapted.pt")
            (stage / "report.html").write_text(render_report(results, curve), encoding="utf-8")
            manifest(stage, {"kind": results["kind"], "seed": config.seed})
            return results
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
