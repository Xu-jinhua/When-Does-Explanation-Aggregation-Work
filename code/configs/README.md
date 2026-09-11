# Experiment configurations

This directory holds every YAML contract consumed by the `xai_ensemble`
pipelines.  Run all commands from the repository's `code/` directory with
`PYTHONPATH=src`.

## Layout

- `simple/` — self-contained "simple" pipeline configurations (one file per
  experiment).  `methods.yaml` is the locked 11+11 CNN/ViT method roster
  referenced via `methods_file`; `assets/` holds the generated BreastMNIST
  manifest and train-image mean used by `quickstart.yaml`.
- `protocols/` — core evaluation-semantics protocol (`core.yaml`) and the
  candidate-grid overlay (`full-matrix.yaml`) for the matrix runner.
- `pilots/` — pilot cost/calibration specifications (e.g. the NOISE
  goodness-of-fit bootstrap budget).

## Which file reproduces which part of the paper

| File | Kind / id | Paper content |
| --- | --- | --- |
| `simple/paper-main.yaml` | experiment `paper-main-v1` | Main result table (Table 1) |
| `simple/paper-naive-ablations.yaml` | ablation `paper-naive-ablations-v1` | NAIVE ablation tables (Tables 2/4/5) |
| `simple/paper-assumptions.yaml` | assumption `paper-assumptions-full-v2` | Three-family q=11 IND / NAIVE / NOISE assumption checks |
| `simple/paper-noise-prefix-sweep.yaml` | sweep `paper-noise-prefix-sweep-v1` | NOISE prefix sweep diagnostic |
| `simple/paper-noise-random-subset.yaml` / `-v2` | studies `paper-noise-random-subset-v1` / `-v2` | Random-subset NOISE studies |
| `simple/paper-noise-random-order-anchored.yaml` | study `paper-noise-random-order-anchored-v1` | Order-anchored NOISE study |
| `simple/paper-relative-robustness.yaml` | study `paper-relative-robustness-v1` | Relative-robustness analysis |
| `simple/paper-noise-generalization-pathmnist-densenet121.yaml` | experiment `noise-generalization-pathmnist-densenet121-v1` | PathMNIST/DenseNet-121 NOISE generalization |
| `simple/paper-noise-generalization-pathmnist-densenet121-prefix.yaml` | sweep `noise-generalization-pathmnist-densenet121-prefix-v1` | Prefix sweep for the generalization cell |
| `simple/paper-noise-generalization-pathmnist-densenet121-assumptions.yaml` | assumption `noise-generalization-pathmnist-densenet121-oracle-noise-v1` | NOISE for the generalization cell |
| `simple/full-matrix.yaml` | experiment `github-complete-results-v2` | Candidate-grid runner and compatibility archive; metric coverage is export-dependent |
| `simple/example.yaml` | experiment `paper-main-v1` | Annotated teaching example of the main configuration |
| `simple/quickstart.yaml` | experiment `quickstart` | Minimal end-to-end demo (see below) |

Derived configurations reference their inputs explicitly (`base_config`,
`assumptions_config`, `noise_prefix_config`, `independent_selector`), so the
dependency order is visible in the files themselves.

## Storage paths and local mode

Each `storage:` block selects a backend by the shape of `remote_root`:

- If the first path segment contains no `:` (e.g. `./runs/<id>/remote`), the
  built-in local filesystem backend is used and the path resolves against the
  working directory.
- To publish to an rclone remote instead, write `remote:path`
  (e.g. `myremote:xai/results`).  The `rclone` binary is discovered on `PATH`
  unless `rclone_binary` overrides it.

All other relative filesystem paths in a config (`manifest_path`,
`mean_path`, `checkpoint_path`, `scratch_root`, `runtime.*`) resolve against
the configuration file's own directory.  `spool_root` stages pending shard
uploads; point it at a tmpfs such as `/dev/shm` and keep
`spool_min_free_gib`/`spool_max_gib` within your machine's budget.

## Filling the placeholders

Paper configurations ship with `/path/to/...` placeholders for the immutable
assets they bind.  Generate each asset with the matching Phase 0 command,
then replace the placeholder:

- `manifest_path` —
  `python -m xai_ensemble.cli phase0 build-manifest --dataset <key> --output <manifest.json> --cache-dir <cache>`
- `mean_path` (plus `mean_key: dataset_mean`) —
  `python -m xai_ensemble.cli phase0 compute-means --dataset <key> --manifest <manifest.json> --model <model_key> --split train --output <dir>/train-image-mean --cache-dir <cache>`
- `checkpoint_path` (with `init_mode: checkpoint`) —
  `python -m xai_ensemble.cli phase0 train ...`; see
  `../scripts/train_example.sh` for a complete, runnable example that goes
  from manifest to a fine-tuned ResNet-18 and prints the exact two lines to
  set in the config.

Every manifest, mean, and checkpoint is content-bound: `simple validate`
rejects an asset whose recorded dataset/model/fingerprint does not match the
configuration.

## Quickstart

`simple/quickstart.yaml` runs one small cell end to end: BreastMNIST
(manifest and train-image mean already generated under `simple/assets/`),
one ImageNet-pretrained ResNet-18 (`init_mode: imagenet1k`, so no training
is needed), the locked eleven-method CNN roster in Phase 1, and a four-method
ensemble evaluated under three aggregation rules in Phase 2.

```bash
python -m xai_ensemble.cli simple validate --config configs/simple/quickstart.yaml
python -m xai_ensemble.cli simple plan     --config configs/simple/quickstart.yaml
python -m xai_ensemble.cli simple run      --config configs/simple/quickstart.yaml
python -m xai_ensemble.cli simple summarize --config configs/simple/quickstart.yaml --table metrics
```

Before the first run, `python scripts/doctor.py` sanity-checks the Python
environment, CUDA runtime, scratch space, and (optionally) rclone.  ViT
methods additionally need the pinned Chefer RelProp checkout:
`python scripts/setup_transformer_explainability.py`.
