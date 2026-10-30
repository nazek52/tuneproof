"""Public command-line interface. ML is imported only when requested."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bundle import prepare, verify
from .data import audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tuneproof", description="Evidence-first fine-tuning lab")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("audit", help="Inspect JSONL without changing it")
    inspect.add_argument("source", type=Path)
    inspect.add_argument("--threshold", type=float, default=0.8)
    prep = commands.add_parser("prepare", help="Export a group-safe Soup data bundle")
    prep.add_argument("source", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    prep.add_argument("--val-fraction", type=float, default=0.2)
    prep.add_argument("--threshold", type=float, default=0.8)
    prep.add_argument("--seed", type=int, default=42)
    experiment = commands.add_parser("experiment", help="Run real CPU pretraining + LoRA + control")
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--pretrain-steps", type=int, default=180)
    experiment.add_argument("--adapter-steps", type=int, default=240)
    experiment.add_argument("--threads", type=int, default=2)
    verify_parser = commands.add_parser("verify", help="Verify all files against a bundle manifest")
    verify_parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            result = audit(args.source, args.threshold).summary()
            code = (
                2
                if result["invalid_rows"]
                or result["conflicting_prompt_lines"]
                or not result["valid_rows"]
                else 0
            )
        elif args.command == "prepare":
            result = prepare(
                args.source,
                args.output,
                base=args.base,
                threshold=args.threshold,
                val_fraction=args.val_fraction,
                seed=args.seed,
            )
            code = 0
        elif args.command == "verify":
            result = verify(args.directory)
            code = 0 if result["ok"] else 2
        else:
            try:
                from .experiment import ExperimentConfig, run_experiment
            except ImportError as exc:
                raise ValueError('install the ML extra first: pip install -e ".[ml]"') from exc
            config = ExperimentConfig(
                seed=args.seed,
                pretrain_steps=args.pretrain_steps,
                adapter_steps=args.adapter_steps,
                threads=args.threads,
            )
            full = run_experiment(args.output, config)
            result = {
                "output": str(args.output),
                "checks": full["checks"],
                "base_nll": full["base"]["response_nll"],
                "lora_nll": full["lora"]["response_nll"],
                "lora_exact_match": full["lora"]["exact_match"],
            }
            code = 0
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return code
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
