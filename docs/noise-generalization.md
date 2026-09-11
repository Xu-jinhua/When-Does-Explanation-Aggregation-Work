# NOISE generalization cell

This cell tests whether the paper's NOISE assumption transfers beyond the four
main dataset/model cells. It uses the official MedMNIST+ PathMNIST 224 release
with a timm DenseNet121 reference model. The cell is intentionally separate
from the main experiments and their artifact namespaces.

The scientific scope is the complete test split, `p=16`, `k=20`, dataset-mean
filling, all eleven CNN explainers, and the five conditions used by the main
experiment: clean, Gaussian 0.15, salt-and-pepper 0.05, speckle 0.15, and the
Sara-style adversarial condition at 2/255. Fidelity orders the eleven
 individual methods on clean test data. The NOISE selector then runs the
existing Spearman/Borda and Kendall/Kemeny distance tests as a historical
comparator. The prefix consumer evaluates every q from 2 through 11 for every
rule and condition. The primary generalization test freezes the
Fidelity-Anchored Top-k Mallows selector from individual clean-F marks and
Borda/Kemeny rank geometry, then compares its one shared q with the
independently computed clean-F oracle q for both geometries.

The shared method catalog still generates p=8, p=14, and p=16 variants for
FeatureAblation and Occlusion in the ordinary Phase 1 scope. This preserves
the established one-pass patch-method behavior and leaves reusable sensitivity
artifacts. The formal NAIVE Phase 2, NOISE selector, and q-prefix sweep
in this cell consume only p=16; the extra two patch variants do not enter the
reported comparison.

The three configurations have separate identities and storage roots:

1. `configs/simple/paper-noise-generalization-pathmnist-densenet121.yaml`
   generates the ordinary NAIVE artifacts and evaluates the formal p=16 cell.
2. `configs/simple/paper-noise-generalization-pathmnist-densenet121-assumptions.yaml`
   sets `science.settings: [oracle-noise]`. It registers only the reusable
   Spearman family, two selections, ten rank tasks, and ten evaluation tasks.
3. `configs/simple/paper-noise-generalization-pathmnist-densenet121-prefix.yaml`
   consumes the completed NAIVE and NOISE artifacts and schedules the
   q sweep.

The NOISE-only assumptions graph deliberately does not train independent
models or generate a second source bank. This is sufficient for the NOISE
hypothesis, which is defined over the complete ordinary NAIVE rank collection.

The transfer decision is fixed before this cell produces results. It is
`strong_support` when the shared q exactly matches both Borda and Kemeny
clean-F oracle q values. It is `support` when both are within one q step and
the maximum absolute clean-F regret is at most 0.005 (half a percentage
point). Every other outcome is `not_supported`. Full quality and robustness
comparisons remain secondary diagnostics and do not change this verdict.

## Preparation

Build the one-time Phase 0 assets — the pinned dataset manifest, the reference
train/validation partitions, the DenseNet121 train mean, and the fine-tuned
reference checkpoint — with the `xai-exp phase0` commands:

```bash
xai-exp phase0 build-manifest \
  --dataset pathmnist \
  --output runs/noise-generalization-pathmnist-densenet121-v1/data/pathmnist.manifest.json

xai-exp phase0 build-partitions \
  --manifest runs/noise-generalization-pathmnist-densenet121-v1/data/pathmnist.manifest.json \
  --kind reference --split train \
  --output runs/noise-generalization-pathmnist-densenet121-v1/data/reference.train.json

xai-exp phase0 compute-means \
  --dataset pathmnist \
  --manifest runs/noise-generalization-pathmnist-densenet121-v1/data/pathmnist.manifest.json \
  --model densenet121 --split train \
  --output runs/noise-generalization-pathmnist-densenet121-v1/artifacts/means.pathmnist.densenet121

xai-exp phase0 train \
  --dataset pathmnist \
  --manifest runs/noise-generalization-pathmnist-densenet121-v1/data/pathmnist.manifest.json \
  --train-partition runs/noise-generalization-pathmnist-densenet121-v1/data/reference.train.json \
  --model densenet121 \
  --recipe timm_finetune --initialization imagenet1k \
  --epochs 100 --device cuda \
  --output runs/noise-generalization-pathmnist-densenet121-v1/models/pathmnist-densenet121/checkpoints
```

The registry key `pathmnist` resolves to the pinned
`medmnist/pathmnist` MedMNIST-3.0.2 224-pixel entry. The official 224 archive
is about 12.6 GB; make sure the download cache and the experiment storage have
capacity before preparation. The checkpoint and mean sidecars must pass
`simple validate` before any queue is started.

## Execution order

Run all commands from `code/` with the package installed.

```bash
xai-exp simple validate \
  --config configs/simple/paper-noise-generalization-pathmnist-densenet121.yaml

xai-exp simple run \
  --config configs/simple/paper-noise-generalization-pathmnist-densenet121.yaml \
  --include-phase2
```

The base queue must be terminal-successful before the two read-only consumers
are started. They use independent SQLite databases, so a rerun is idempotent
and cannot alter the base NAIVE queue.

```bash
xai-exp simple assumptions validate \
  --config configs/simple/paper-noise-generalization-pathmnist-densenet121-assumptions.yaml

xai-exp simple assumptions readiness \
  --config configs/simple/paper-noise-generalization-pathmnist-densenet121-assumptions.yaml

xai-exp simple assumptions run \
  --config configs/simple/paper-noise-generalization-pathmnist-densenet121-assumptions.yaml

xai-exp simple noise-prefix readiness \
  --config configs/simple/paper-noise-generalization-pathmnist-densenet121-prefix.yaml

xai-exp simple noise-prefix run \
  --config configs/simple/paper-noise-generalization-pathmnist-densenet121-prefix.yaml
```

After the base, assumptions, and q-sweep queues are terminal-successful, first
write the base and historical NOISE summaries:

```bash
xai-exp simple summarize \
  --config configs/simple/paper-noise-generalization-pathmnist-densenet121.yaml \
  --table table1 \
  --manifest-source local \
  --output-directory results/simple/noise-generalization-pathmnist-densenet121/table1

xai-exp simple assumptions summarize \
  --config configs/simple/paper-noise-generalization-pathmnist-densenet121-assumptions.yaml \
  --scope noise \
  --output-directory results/simple/noise-generalization-pathmnist-densenet121/oracle-noise

xai-exp simple noise-prefix summarize \
  --config configs/simple/paper-noise-generalization-pathmnist-densenet121-prefix.yaml \
  --output-directory results/simple/noise-generalization-pathmnist-densenet121/q-sweep
```

## Frozen selector and transfer report

The primary generalization test freezes its selector before any q-level
aggregate metric is read: the selector consumes only individual clean-F marks
and rank-only candidate centers, and produces a
`simple-dual-geometry-noise-selector-v1` selector document whose frozen
identity is SHA-256-verified on load
(`xai_ensemble.simple.noise_prefix.dual_geometry`). In the matrix
pipeline this step runs as dedicated `selector-cell` / `merge-selector` jobs
(see [`full-matrix.md`](full-matrix.md)); for this standalone cell the same
library post-processing is applied to the completed base and prefix artifacts
summarized above.

The report retains every per-condition/per-rule q value, the historical
NOISE fits, the new shared q, and every endpoint comparison. In
particular, the selection-validation rows record the selected q, the
Borda/Kemeny clean-F oracle q, exact and within-one indicators, selected and
oracle Fidelity, and Fidelity regret. Together these form the complete
evidence for deciding whether the new NOISE assumption transfers to this
cell; no IND source bank is required for that question.

Only the small provenance-rich JSON/CSV reports belong in a results Git
repository. Datasets, checkpoints, scheduler databases, full attributions,
and per-sample Phase 2 shards remain in their configured storage namespaces.
The same three-config layout and generic selector/report code can be reused
when the candidate grid is expanded.
