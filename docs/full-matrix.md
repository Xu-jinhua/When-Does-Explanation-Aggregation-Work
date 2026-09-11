# Matrix planning and compatibility archive

This document records the experiment definition used to plan the broad
dataset/model study and to archive compatibility checks. It is a runbook, not
a claim that every planned cell has completed quality or robustness metrics:
`validate` and `plan` are read-only, and execution is gated behind an explicit
flag (see [Commands](#commands)).

## Scope

The candidate grid contains 14 datasets and eight classifier architectures.

- Datasets: ImageNet100, Food101, Places365, PathMNIST, DermaMNIST, OCTMNIST,
  PneumoniaMNIST, RetinaMNIST, BreastMNIST, BloodMNIST, TissueMNIST,
  OrganAMNIST, OrganCMNIST, and OrganSMNIST.
- Models: ResNet18, ResNet50, DenseNet121, EfficientNet-B0, MobileNetV3-Large,
  ViT-B/16, DeiT-B/16, and Swin-B.

All MedMNIST inputs use the official MedMNIST+ 224-pixel files. Grayscale
datasets use `as_rgb=True`, which replicates the channel to RGB before
preprocessing. Food101 and Places365 use the pinned composite Hugging Face
views in `xai_ensemble.data.specs`; their deterministic train/validation
construction belongs to the dataset identity.

Each non-blocked reference classifier is ImageNet-1K initialized and
fine-tuned for 100 epochs in BF16. It produces a full-reference checkpoint, a
preprocessing-specific train mean, and a deterministic validation sample set.
Every member of the architecture's 11-method roster must pass the
real-checkpoint compatibility gate. A failed method blocks the cell and is
recorded in the active-cell manifest; it is never silently removed from the
roster. This strict rule includes Swin-B.

Before training, the scheduler performs a static provider preflight. It writes
a terminal blocked gate only when the pinned provider has no scientifically
equivalent constructor for a required method-model pair. In the current pinned
checkout, PartialLRP and FullLRP lack a DeiT-B/16 constructor, and
CheferTransformerAttribution, PartialLRP, and FullLRP lack a Swin-B
constructor. Those 28 cells — every DeiT-B/16 and every Swin-B cell — are
recorded as blocked without wasting 100-epoch training jobs. Every remaining
cell still requires the real-checkpoint gate; a later generic-method failure
also blocks its complete cell.

CNNs use Saliency, InputXGradient, IntegratedGradients, GuidedBackprop,
Deconvolution, FeatureAblation, Occlusion, DeepLift, GradientShap,
DeepLiftShap, and LRP. Transformer models use Saliency, InputXGradient,
IntegratedGradients, FeatureAblation, Occlusion, GradientShap,
CheferTransformerAttribution, PartialLRP, FullLRP, GradientAttentionRollout,
and AttentionGradCAM.

The planned release evaluates only the `p=16` FeatureAblation and Occlusion
variants. The shared method catalog retains `p=8` and `p=14` for the already
separate ablation studies, but the candidate-grid runner neither generates nor
evaluates those variants. The primary release is fixed to the test split,
`p=16`, `k=20`, and train-split `dataset_mean` filling.

## Candidate-grid status

The current configuration enumerates the candidate grid. Static provider
preflight records constructor-blocked architecture/dataset combinations, and
the real-checkpoint gate records compatibility results for the remaining
combinations. `validate` and `plan` are read-only; neither creates the SQLite
queue nor launches a GPU worker. The generated LaTeX archive includes only
completed exports and explicitly labels compatibility-only records as
diagnostics.

## Data flow

### Phase 0

Phase 0 builds content-pinned dataset manifests, deterministic reference
train/validation partitions, and a train mean for each dataset/model
preprocessing contract. Training writes resumable scratch checkpoints,
verifies the inference checkpoint SHA-256, and atomically publishes the
completed checkpoint to the artifact store. Scratch is removed only after the
published checkpoint and sidecar validate.

The compatibility gate reloads the exact checkpoint and mean artifact, runs
all 11 final `methods.yaml` configurations on real validation samples, and
writes an immutable gate record. Its candidate digest includes the final
parameters and the `p=16` patch variants, so it cannot be satisfied by older
protocol defaults. Only passed cells appear in generated component
configurations.

### Phase 1

For every active cell and condition, the reference model first predicts the
clean input in FP32. Those full-reference predictions, logits, and targets are
saved, and the clean prediction is passed to the explainer. The explained
target is not the ground-truth label or a perturbed argmax.

Phase 1 writes 512-sample safetensors shards containing attribution values,
sample identifiers, full-reference outputs, targets, and manifest provenance.
Attribution remains channel-preserving where later reduction needs it.
Rank-ready sidecars derive the paper patch score from absolute attribution and
canonical patch/channel reduction. Payload shards publish first and manifests
publish last.

The conditions are clean, Gaussian `0.15`, salt-and-pepper `0.05`, speckle
`0.15`, and saved Sara-style per-sample adversarial corruption at
`epsilon=2/255`. Natural corruptions are deterministic factories. The
expensive adversarial inputs are persisted for reproducibility and reuse.

### Phase 2

Phase 2 consumes aligned Phase 1 artifacts, computes strict patch ranks,
builds aggregation ranks on GPU, and evaluates clean, removed-top-k, and
retained-top-k inputs with the reference model. `clean` is the same
preprocessed original. `removed` fills top-ranked patches with the dataset
mean. `retained` fills their complement. The Phase 1 clean output is reused
for target identity, but masked inputs require new forwards.

The matrix retains individual methods plus SimpleAvg, Borda, Kemeny, RRF, and
Schulze. SimpleAvg follows its fixed per-sample/per-method normalization
contract, while rank rules use the p=16 rank bank. Phase 2 output retains raw
metric ingredients and traces needed for table summaries and signed robustness
without rerunning attribution.

### NOISE and IND

NOISE evaluates every Fidelity-ordered prefix `q=2..11` from immutable clean
ranks. The independent-geometry Fidelity-Anchored Top-k Mallows selector
freezes Spearman/Borda and Kendall/Kemeny q values independently and does not
read q-level aggregate metrics during selection. The frozen selector drives
all five aggregation rules.

IND trains eleven disjoint source models per cell and assigns the eleven
explanation methods through a deterministic bijection. The corresponding
NAIVE comparison and NOISE selector are separate consumers of the immutable
artifacts. Historical runs may use a smaller source subset; those exports are
kept as historical tables and are not silently presented as a complete IND
replication.

## Catalog, planner, and selector

The `xai_ensemble.simple.full_matrix` package separates three concerns:

- the **catalog** fixes the 14-by-8 cell grid, the per-architecture method
  rosters, and the static provider preflight that records blocked cells;
- the **planner** expands the catalog and the locked protocol into the staged
  job DAG (assets, gates, NAIVE, NOISE sweep, IND, coverage), reusing
  the component experiments' immutable task identities so existing artifacts
  make resubmitted jobs succeed without overwrite;
- the **selector** materializes the frozen NOISE q choices per cell
  (`selector-cell`) and merges them (`merge-selector`) without reading q-level
  aggregate metrics.

## Scheduler and storage

`simple full-matrix run` owns one SQLite queue. It extends the DAG only after
the prior barrier succeeds:

1. assets and compatibility gates;
2. frozen active-cell configs;
3. NAIVE profiling, Phase 1, and Phase 2;
4. rank-ready preparation and the NOISE q-sweep;
5. selector partials, merge, and selected-NOISE summary;
6. IND and NAIVE;
7. coverage summary.

Component schedulers are never started. Their idempotent jobs are namespaced
into this one queue and retain their existing artifact identities. Failed jobs
retry to the configured limit and then block dependents. Existing valid
artifacts make resubmitted jobs succeed without overwrite.

GPU admission uses live NVML free memory, active reservations, and a fixed 5%
headroom. High-memory profile, FeatureAblation, and Occlusion jobs preserve
exclusive placement. CPU manifest, selector, summary, and barrier jobs run
with `CUDA_VISIBLE_DEVICES=""`, are capped by `max_cpu_jobs`, and reserve no
GPU. A bounded tmpfs area holds scratch; immutable artifacts, reference
checkpoints, and the large provider dataset cache live in the configured
artifact storage rather than the constrained local disk. Local filesystem mode
and rclone remotes are both supported; see
[`experiment-guide.md`](experiment-guide.md#storage-configuration).

## Commands

Run from `code/` with the package installed:

```bash
xai-exp simple full-matrix validate --config configs/simple/full-matrix.yaml
xai-exp simple full-matrix plan     --config configs/simple/full-matrix.yaml
```

The formal run is explicitly gated:

```bash
xai-exp simple full-matrix run \
  --config configs/simple/full-matrix.yaml \
  --confirm-full-matrix
```

Use `status` for the single queue and `retry --job-id <id>` only after a code
or environment repair. Validation, planning, testing, and importing the
package never start the candidate grid.
