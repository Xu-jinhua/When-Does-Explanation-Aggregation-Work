# Experiment code

This directory contains the implementation of the experiments in *When Does
Aggregating Explanations Work?*: a reproducible Phase 0/1/2 stack (data
identity and model training, explanation generation, rank aggregation and
evaluation) plus the paper-first two-stage pipeline that runs the published
experiments end to end. The package is `xai_ensemble`; its entry point is the
`xai-exp` command.

The code deliberately separates scientific decisions from runtime decisions:
method rosters, patch sizes, aggregation rules, and evaluation semantics are
locked in versioned YAML configurations, while batch sizes, GPU placement, and
I/O concurrency are measured or configured independently and never enter the
scientific identity of an artifact.

## Installation

Python 3.11 or 3.12 is required (`>=3.11,<3.13`). From this directory:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[gpu,dev]"
```

Extras:

- `gpu` — the experiment runtime: PyTorch, torchvision, timm, Captum, einops,
  safetensors, Hugging Face `datasets`, pyarrow/pandas, and MedMNIST. This
  extra registers the `xai-exp simple` and `xai-exp phase0` command groups;
  without it only the CPU-only `xai-exp protocol` group is available.
- `dev` — pytest, pytest-cov, and ruff.

A CUDA GPU is required to *run* Phase 1/Phase 2 jobs and the memory profiles,
but not to install the package or to use the read-only `validate`/`plan`
commands. ViT LRP-family methods additionally use the pinned third-party
snapshot under [`vendor/`](vendor/); no extra installation is needed for it.

## Command-line overview

All commands accept `--help`. The top-level groups are:

```text
xai-exp protocol   Validate and snapshot a machine-readable protocol
xai-exp phase0     Prepare data and train models
xai-exp simple     Run the paper-first two-stage experiment
```

### `xai-exp phase0` — assets

| Subcommand | Purpose |
|:--|:--|
| `build-manifest` | Scan a pinned dataset into a content-hashed manifest (stable sample IDs, labels, splits, per-record hashes, fingerprint) |
| `build-partitions` | Build class-stratified `ind` / `overlap` / `reference` training partitions from a manifest |
| `train` | Train or resume one source/reference model (timm fine-tune, subset-logit, or random-scratch recipes); writes checksummed inference checkpoints with full metadata sidecars |
| `compute-means` | Compute the immutable pixelwise train-split dataset/class mean images used for mask filling and baselines |

### `xai-exp simple` — the paper pipeline

Read-only inspection (no GPU, no job submission):

| Subcommand | Purpose |
|:--|:--|
| `validate` | Check the configuration, all dataset manifests, train-mean artifacts, checkpoint sidecars and SHA-256 values, and the vendored RelProp runtime |
| `plan` | Print (or write with `--output`) the deterministic profiles, adversarial datasets, and Phase 1/Phase 2 task graph |
| `status` | Report profile coverage and the SQLite queue state |

Execution (GPU):

| Subcommand | Purpose |
|:--|:--|
| `profile` / `phase2-profile` | Run one missing explanation-batch or Phase-2 memory profile (`--profile-id`) |
| `adversarial` | Generate one immutable sharded adversarial dataset (`--task-id`) |
| `phase1` / `phase2` | Run one explanation-generation or aggregation/evaluation task (`--task-id`) |
| `run` | Drive the queue: profiles first, then Phase 1; add `--include-phase2` to continue through aggregation and evaluation |
| `retry` | Requeue explicitly named failed jobs (`--job-id`) after a repair |

Post-processing (CPU):

| Subcommand | Purpose |
|:--|:--|
| `summarize` | Write the deterministic Table 1 summary (JSON/CSV/TeX) from completed Phase 2 artifacts |
| `effective-robustness` | Offline Effective Robustness from completed quality summaries |
| `effective-robustness-noise-bootstrap` | Paired-bootstrap NOISE-vs-NAIVE ER significance test |
| `quality-retention` | Normalized absolute quality, clean retention, and geometric mean |
| `adversarial-pilot` | Profile and validate the Sara-style explanation attack before formal use |

Sub-experiment groups (each with its own `validate`/`plan`/`run`/`status`/
`retry` and task-level commands; see `--help`):

| Group | Experiment |
|:--|:--|
| `simple ablation` | Isolated NAIVE ablations (k, fill, noise strength) over immutable main artifacts |
| `simple assumptions` | IND, NAIVE, and NOISE assumption checks (plus the `table-priority-*` IND queue) |
| `simple noise-prefix` | Every Fidelity-ordered NOISE prefix q = 2..11 |
| `simple noise-subset` | Random NOISE-consistent subsets and the random-order anchored control |
| `simple relative-robustness` | Fixed random-mask controls and null-anchored R_rel |
| `simple full-matrix` | Matrix planning and compatibility archive (gated by `--confirm-full-matrix`) |

The scheduler is the normal entry point (`run`); individual task IDs shown by
`plan` may also be executed directly with the matching subcommand. Queue state
lives in the configured SQLite database; verified artifacts make every retry
idempotent.

## The two-stage pipeline and its artifacts

**Phase 1** stores one immutable artifact per
dataset/model/split/condition/method variant. Each artifact is a set of
safetensors shards (512 samples per shard by default) plus a `manifest.json`
that is published only after every shard has been uploaded and verified.
Shard schema version 2:

```text
indices       int64   [N]
labels        int64   [N]
predictions   int64   [N]
logits        float32 [N, num_classes]
targets       int64   [N]
attributions  float32 [N, C, H, W]
```

Everything is FP32. A clean job explains the model's own FP32 prediction; a
perturbed job explains the matching *clean* FP32 prediction while storing the
perturbed input's current prediction and full logits. The writer rejects a
shard unless `predictions == argmax(logits)` for every row, and Phase 2
re-checks the invariant after download. The attribution is stored complete
and signed — no absolute value, channel reduction, patching, or ranking
happens in Phase 1. Manifests additionally record machine-readable
`target_policy` and `model_output_source` fields, and are immutable once
published.

**Phase 2** loads one aligned shard from every requested method, computes each
method's patch score as `mean(abs(attribution))` over channels and pixels
inside each `p × p` patch, and turns descending scores into strict ranks with
deterministic row-major tie breaking. SimpleAvg follows its own path
(per-method absolute/channel-mean map, spatial normalization locked to
`minmax`, mean across methods, then patch scores); Borda, RRF, Kemeny-Young,
and Schulze consume the per-method rank ballots. The consensus stage runs as
batched Torch kernels on the task's GPU; Kemeny uses Borda as its single start
with at most 1024 strict-improvement insertion moves per sample. For each rule
and sample, the top-`k` patches define a `removed` and a `retained` image
(train-split mean fill, preprocessing applied after masking), and the reference
model forwards only those two variants — the Phase 1 prediction is reused as
the unmasked prediction. Per-sample sufficient statistics (every rank, every
removed/retained prediction) are preserved in the Phase 2 shards, so tables,
bootstraps, and stratified post-processing never need another model forward.

See [`../docs/experiment-guide.md`](../docs/experiment-guide.md) for the full
scientific contract, metric definitions, and per-experiment command chains.

## Storage: local mode and rclone remotes

`storage.remote_root` selects the artifact backend. If its first path segment
contains no `:` the built-in **local filesystem mode** is used: payloads are
published atomically (temporary file plus rename), existing files are
SHA-256-checked and never replaced, and the manifest is written last. If the
first segment contains a `:` (for example `remote:bucket/path`), publication
goes through **rclone** (`copyto --immutable`) with size and SHA-256
verification from backend metadata where available, a streaming verification
fallback otherwise, and an immutable per-file receipt.

In both modes a bounded RAM-backed spool (default
`/dev/shm/xai-simple/<experiment_id>`, at most 64 GiB pending with a 32 GiB
free-space floor) decouples GPU generation from publication and provides
explicit backpressure. `storage.rclone_binary` is optional and defaults to a
`PATH` lookup; note that `simple validate` checks that the binary exists even
when the local backend is configured.

Execution-only knobs (worker counts, spool and prefetch budgets, telemetry
interval) can be overridden through `XAI_SIMPLE_*` environment variables
without changing the scientific or scheduler identity of a run; see
`src/xai_ensemble/simple/config.py` for the full list.

## Layout

```text
code/
├── configs/
│   ├── protocols/         # Machine-readable dataset/training protocols
│   ├── pilots/            # Small experiments with explicit pass criteria
│   └── simple/            # One versioned YAML per published experiment
├── scripts/               # setup, audits, examples, and result-to-LaTeX export
├── src/xai_ensemble/
│   ├── core/              # Hashing, atomic I/O, paths, manifests, provenance
│   ├── data/              # Pinned dataset registry, manifests, partitions
│   ├── phase0/            # Model registry, training, checkpoints, statistics
│   ├── phase1/            # Explainer registry, ViT RelProp bridge, compatibility gate
│   ├── phase2/            # Mask-game evaluator, metrics, GPU aggregation, bootstrap
│   └── simple/            # Two-stage pipeline and all sub-experiments
├── tests/                 # Unit tests and optional CUDA integration tests
└── vendor/                # Pinned Transformer-Explainability snapshot (MIT)
```

## Testing

```bash
pytest tests
```

The suite is CPU-only by default; CUDA-dependent tests (for example the
Torch/NumPy aggregation equivalence checks) skip themselves when no GPU is
visible, and `pyproject.toml` declares `gpu` / `network` / `slow` markers for
optional integration tests. Some tests write sizeable fixtures below the
system temp directory — on hosts with a small `/tmp`, point `TMPDIR` at a
larger partition first.

## Vendored dependency

[`vendor/Transformer-Explainability/`](vendor/) is a pinned snapshot
(revision `c3e578f76b954e8528afeaaee26de3f07e3fe559`) of Hila Chefer's
Transformer-Explainability implementation, providing the genuine ViT `LRP`,
`PartialLRP`, and `FullLRP` methods. Every imported file is verified against a
SHA-256 allowlist before model construction; a wrong revision or modified file
fails closed. The upstream project is MIT licensed and its license is retained
both as `vendor/Transformer-Explainability.LICENSE` and inside the snapshot.
To re-verify the snapshot, or to fetch it fresh into a development checkout:

```bash
python scripts/setup_transformer_explainability.py --verify-only
```

## License

MIT (see [`../LICENSE`](../LICENSE)). The vendored snapshot remains under its
own MIT license.
