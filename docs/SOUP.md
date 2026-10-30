# Soup integration contract

TuneProof is independent of Soup. No upstream PR or affiliation is claimed.

## What is tested

The optional `soup` extra pins `soup-cli==0.77.1`. The contract test imports its
actual `SoupConfig` model and validates the generated YAML. It does not replace
Soup with a mock and does not download a model.

```bash
python -m pip install -e ".[soup,dev]"
python -m pytest tests/test_soup.py -v
```

Upstream references checked for this implementation:

- [schema.py](https://github.com/MakazhanAlpamys/Soup/blob/main/src/soup_cli/config/schema.py)
- [Data documentation](https://github.com/MakazhanAlpamys/Soup/blob/main/docs/data.md)
- [README](https://github.com/MakazhanAlpamys/Soup)

`data.eval` is not an accepted field in that schema. TuneProof writes only the
training path into `soup.yaml`, sets `data.val_split: 0.0`, and preserves its
independent holdout as `validation.jsonl`. This prevents Soup from accidentally
creating a different split. It does **not** automatically evaluate that holdout.

## Optional real Soup training

Requires Soup's training extras and sufficient hardware. This step was not run
as part of the checked-in tiny custom-model experiment. It downloads a separate
pretrained model; training duration and memory depend on the selected base.

```bash
python -m pip install "soup-cli[train]==0.77.1"
python -m tuneproof prepare examples/clean.jsonl --output runs/soup-data
cd runs/soup-data
soup doctor --config soup.yaml
soup train --config soup.yaml --yes
```

Run from inside the bundle: paths in its YAML are relative. Use Soup's documented
evaluation/inference commands to evaluate its trained model on the separate
holdout, applying the same rendering and response mask. Do not report tiny-model
metrics as results from this unexecuted Soup training.

## Why this is useful alongside Soup

Soup already provides extensive data cleaning, decontamination, evaluation and
training tools. TuneProof is a small inspectable learning lab, emphasizing
transitive prompt grouping, artifact provenance and an explicit matched optimizer
control. It is not a claim that Soup lacks those broader capabilities.
