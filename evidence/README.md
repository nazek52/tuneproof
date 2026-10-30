# Executed evidence

These files are outputs of real local commands, not example numbers.

## Validation

- **84 tests passed, 0 failed, 0 skipped** on Python 3.12.
- **98% rounded statement coverage**; exact totals in `validation.json`.
- Ruff check and format check passed.
- Actual `soup-cli==0.77.1` schema accepted the exported configuration.
- Both artifact manifests passed SHA-256 verification.

The local run is not evidence that remote GitHub Actions has passed. Its live
status must be checked separately.

## Reference experiment

```bash
python -m tuneproof experiment --output runs/reproduce
python -m tuneproof verify evidence/reference
python -m tuneproof verify evidence/soup-bundle
```

| Checkpoint | Response NLL | Exact compact format | Routing label match |
| --- | ---: | ---: | ---: |
| Base | 3.610385 | 0 / 32 | 32 / 32 |
| LoRA | 0.001191 | 32 / 32 | 32 / 32 |
| Zero LR | 3.610385 | 0 / 32 | 32 / 32 |

**Interpretation:** LoRA learned the compact format. This run does not demonstrate
an improvement in choosing the correct category: the base already did that.
The training and holdout share templates and differ in ticket IDs, so this is an
easy in-distribution experiment. No real-world accuracy claim follows.

92,620 total parameters; 7,372 adapter parameters (7.96%). Frozen weights stayed
unchanged; the zero-LR adapter stayed unchanged and its logits exactly matched
the base. Pretraining used 180 steps; each adapter phase used 240 steps. Seed 42.

`reference/` contains the full measured report, curves, per-example predictions,
datasets, base/adapted checkpoints and hash manifest. Open `reference/report.html`
after cloning; GitHub displays HTML source rather than executing it. Run time is
recorded in `results.json` and is machine-specific. Checkpoints were generated
locally; load only trusted checkpoints with `weights_only=True`.

`soup-bundle/` contains the actual prepared data/configuration. It has 8 training
rows and 2 validation rows; the exact duplicate stays in one component. The
validation file is not automatically consumed by Soup's training YAML.

## Reproduction limits

The implementation and seed are available, but Torch kernels and versions can
change numeric results across environments. `validation.json` records the tested
package versions. Manifests validate exact stored bytes, not the truth of an ML
claim or the authenticity of a publisher. No large pretrained model was trained
with Soup in this validation.
