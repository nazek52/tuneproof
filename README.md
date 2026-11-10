<a id="tuneproof"></a>

<p>
  <img src="docs/assets/project-header.svg" alt="TuneProof — Fine-tuning with evidence." width="100%">
</p>

[![Checks](https://github.com/nazek52/tuneproof/actions/workflows/ci.yml/badge.svg)](https://github.com/nazek52/tuneproof/actions/workflows/ci.yml)

A reproducible LoRA experiment with a byte-level causal transformer, leakage-aware dataset preparation, held-out evaluation, and a matched zero-learning-rate control. Run it locally on CPU and inspect the reports, checkpoints, and integrity manifests.

Independent educational project using a tiny synthetic task; not a pretrained-LLM benchmark or an affiliation with Soup.

**Python · PyTorch · LoRA · evaluation**

[Quick start](#quick-start) · [Architecture](docs/ARCHITECTURE.md) · [Results & evidence](evidence/README.md) · [Русский разбор](docs/DEFENSE_RU.md)

---

## Quick start

Python 3.11+ in a virtual environment. Commands work in PowerShell and bash.

```bash
git clone https://github.com/nazek52/tuneproof.git
cd tuneproof
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[ml,dev]"
python -m tuneproof experiment --output runs/demo
python -m tuneproof verify runs/demo
```

Open `runs/demo/report.html` in a browser. It contains measured curves, all held-out
predictions, trainable parameter counts, control checks and limitations. Outputs include
base/adapted checkpoints, JSON evidence and a SHA-256 manifest. Existing output directories
are refused: choose a fresh path for every run.

For CPU-only installation with smaller downloads, install PyTorch first:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[ml,dev]"
```

## Dataset audit and Soup handoff

The lightweight data tools do not import PyTorch:

```bash
python -m pip install -e .
python -m tuneproof audit examples/dirty.jsonl
# Intentionally exits 2: invalid rows / conflicting answers.
python -m tuneproof prepare examples/clean.jsonl --output runs/soup-data
python -m tuneproof verify runs/soup-data
```

Supports Alpaca and single-turn ChatML with optional system message. Exact prompts
and lexical near-duplicates are assigned to connected components before splitting.
Conflicting answers for identical canonical prompts block preparation. Input files
are never modified. Limits: 10 MiB / 2,000 nonblank rows; comparison is O(n²).

The bundle contains `train.jsonl`, `validation.jsonl`, `soup.yaml`, `audit.json` and
`manifest.json`. Soup config fields were checked against the upstream schema; see
[the integration notes](docs/SOUP.md). `val_split: 0` prevents a new row-level split.
**The holdout is separate: this YAML does not automatically evaluate it in Soup.**
The tiny experiment checkpoints are custom PyTorch state dicts, not PEFT adapters.

## Why the negative control matters

The base model learns verbose ticket labels. LoRA adapts it to compact labels using
only low-rank matrices; the pretrained weights stay frozen. A second adapter sees
the identical minibatches with LR=0. Its weights and logits must stay identical to
the base. Plausible-looking loss logs alone would not establish this.

Held-out response NLL uses UTF-8 byte tokens, including EOS; greedy exact match requires
the exact label and formatting. Training and evaluation use separate ticket IDs but
the same lexical templates. A low score proves only performance on this easy toy task.
The report explicitly shows the base's generated text so formatting improvement cannot
be misrepresented as newly acquired semantic knowledge.

## Architecture and tests

[Architecture](docs/ARCHITECTURE.md) was committed before implementation.
[Interview walkthrough (Russian)](docs/DEFENSE_RU.md) explains the code and includes
hands-on change requests. [Evidence](evidence/README.md) records the actual validation.

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=tuneproof --cov-report=term-missing
```

Tests cover JSON/schema failures, Unicode duplicates, transitive components, splitting,
artifact tampering, CLI flows, causal attention, response masking, LoRA identity,
gradient updates, frozen base and the real training/control experiment. GitHub Actions
runs on Python 3.11/3.12. No paid service or credentials are needed.

## Scope and provenance

Independent educational implementation developed with AI assistance. Source code,
validation evidence, and a Russian walkthrough are included so the experiment can be
inspected and reproduced.

No semantic decontamination, privacy certification, distributed training, production
service, large-model result or guaranteed improvement. Manifests detect accidental
changes, not authenticity. Results vary by PyTorch/platform; seed reproducibility
is scoped to the same software/hardware environment. Checkpoints must not be loaded
from untrusted sources; use `torch.load(..., weights_only=True)`.

## References

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [PyTorch causal attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Soup](https://github.com/MakazhanAlpamys/Soup)

MIT license. Synthetic demonstration data are generated by this project; no private
user data or third-party model weights are included.
