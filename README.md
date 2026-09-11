# When Does Aggregating Explanations Work?

Jinhua Xu, Davide Anguita, Fabio Roli, Jing Yuan, and Luca Oneto

## Abstract

Post-hoc explanation methods for image classifiers often produce inconsistent
results, particularly under data perturbations, model variations, or when
different explanation techniques are applied. Ensembling multiple explanations
has therefore emerged as a common strategy to mitigate such disagreement.
However, existing approaches remain largely empirical and offer limited
theoretical insight into the conditions under which ensemble methods improve
reliability or, conversely, fail. In this paper, we study the aggregation of
image explanations derived from patch-based saliency maps. We first rank image
patches according to their estimated importance and then combine these rankings
using different rank-aggregation strategies. Through experiments on multiple
datasets, model architectures, post-hoc explanation methods, and
rank-aggregation strategies, we show that the effectiveness of aggregation
methods depends on how well two assumptions are met: independence among the
aggregated explanations and compatibility of the explanation noise with a
distance-based rank-noise model.

## Results

The experimental results are organized as dataset-level LaTeX reports. Each
report collects the released model and aggregation-setting results for one
dataset. PDF compilation is intentionally left to the reader's LaTeX
environment.

| Dataset | Models represented in the archive | LaTeX source |
|:--|:--|:--|
| ImageNet | DenseNet-121, ResNet-18, ViT-B/16 | [source](results/latex/by-dataset/imagenet/) |
| DermaMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [source](results/latex/by-dataset/dermamnist/) |
| PathMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [source](results/latex/by-dataset/pathmnist/) |
| BloodMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [source](results/latex/by-dataset/bloodmnist/) |
| BreastMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [source](results/latex/by-dataset/breastmnist/) |
| Food101 | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18 | [source](results/latex/by-dataset/food101/) |
| OCTMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [source](results/latex/by-dataset/octmnist/) |
| OrganAMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [source](results/latex/by-dataset/organamnist/) |
| OrganCMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [source](results/latex/by-dataset/organcmnist/) |
| OrganSMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [source](results/latex/by-dataset/organsmnist/) |
| Places365 | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [source](results/latex/by-dataset/places365/) |
| PneumoniaMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [source](results/latex/by-dataset/pneumoniamnist/) |
| RetinaMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [source](results/latex/by-dataset/retinamnist/) |
| TissueMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [source](results/latex/by-dataset/tissuemnist/) |

The generated reports currently contain 705 non-empty tables covering 14
datasets and only the dataset/model rows present in the released source
files. Results from current snapshots, historical experiment versions, and
real-checkpoint compatibility diagnostics are retained together. Compatibility
tables are explicitly archival checks, not completed Phase 2 quality or
robustness metrics; the reports do not claim complete coverage for every
possible dataset/model combination. Counts, source paths, versions, and
empty-source records are listed in
[`results/latex/README.md`](results/latex/README.md) and
[`results/latex/RESULT_MANIFEST.json`](results/latex/RESULT_MANIFEST.json).

## Repository layout

```text
.
├── results/
│   ├── pdf/                 # Published result reports
│   └── latex/               # LaTeX sources for the published PDFs
├── code/                    # Experiment implementation (see code/README.md)
│   ├── src/xai_ensemble/    # Python package: Phase 0/1/2 and the paper pipeline
│   ├── configs/             # Experiment configurations (protocols, pilots, simple)
│   ├── scripts/             # Setup, training-example, and audit utilities
│   ├── tests/               # Unit and CUDA integration tests
│   └── vendor/              # Pinned third-party ViT explanation provider
└── docs/                    # Experiment guides (pipeline, full matrix, NOISE transfer)
```

## Code

The experiment implementation lives in [`code/`](code/) and installs as the
`xai_ensemble` Python package with the `xai-exp` command-line entry point.
See [`code/README.md`](code/README.md) for the full package reference and
[`docs/`](docs/) for the experiment guides.

### Installation

Python 3.11 or 3.12 is required (`>=3.11,<3.13`). From a fresh virtual
environment:

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -e ".[gpu,dev]"
```

The `gpu` extra pulls in the experiment runtime (PyTorch, torchvision, timm,
Captum, safetensors, MedMNIST, Hugging Face datasets) and registers the
`xai-exp simple` and `xai-exp phase0` command groups; the `dev` extra adds
pytest and ruff. A minimal `pip install -e .` provides only the CPU-only
`xai-exp protocol` group. A CUDA GPU is needed to *run* experiments, but not
to install, validate, or plan them. Phase 1 also verifies a pinned snapshot of
Hila Chefer's `Transformer-Explainability` code, vendored under
[`code/vendor/`](code/vendor/) (MIT licensed, license retained).

### Quickstart

[`code/configs/simple/quickstart.yaml`](code/configs/simple/quickstart.yaml)
is a self-contained demonstration: one small public dataset
(BreastMNIST-224), one ImageNet-initialized ResNet-18 (pretrained weights are
downloaded automatically by timm, so no private checkpoint is needed), and
committed demo assets (dataset manifest and train-split mean). Phase 1 runs
the locked eleven-method CNN explanation roster; Phase 2 then evaluates an
ensemble of four of those methods under three aggregation rules. It uses
local filesystem storage and writes everything under
`code/configs/simple/runs/quickstart/` (paths resolve relative to the
configuration file).

```bash
cd code
xai-exp simple validate  --config configs/simple/quickstart.yaml
xai-exp simple plan      --config configs/simple/quickstart.yaml
xai-exp simple run       --config configs/simple/quickstart.yaml --include-phase2
xai-exp simple summarize --config configs/simple/quickstart.yaml --table metrics
```

`validate` and `plan` are read-only and need no GPU; they check the
configuration, the committed assets, and the runtime contract, then print the
deterministic task graph. Note that `validate` also requires an `rclone`
executable on `PATH` even in local storage mode. `run` executes the two-stage
pipeline on one GPU: without `--include-phase2` it stops after the
explanation stage. `summarize` then builds the table-ready summary from the
completed evaluation artifacts; the default `--table table1` implements the
paper's fixed condition layout, while `--table metrics` (used above) is a
condition-agnostic mode that works for any configuration, including this
clean-only demo.

### Reproducing the paper

Every paper experiment is a versioned YAML configuration under
[`code/configs/simple/`](code/configs/simple/), executed through one
`xai-exp` command group:

| Paper artifact | Configuration | Command group |
|:--|:--|:--|
| Table 1 (main NAIVE results) | `paper-main.yaml` | `xai-exp simple run --include-phase2`, then `xai-exp simple summarize` |
| Tables 2, 4, 5 (k, fill, noise-strength ablations) | `paper-naive-ablations.yaml` | `xai-exp simple ablation` |
| IND, NAIVE, and NOISE assumption tables | `paper-assumptions.yaml` | `xai-exp simple assumptions` |
| NOISE Fidelity-prefix sweep (q = 2..11) | `paper-noise-prefix-sweep.yaml` | `xai-exp simple noise-prefix` |
| NOISE generalization cell (PathMNIST / DenseNet-121) | `paper-noise-generalization-pathmnist-densenet121{,-assumptions,-prefix}.yaml` | `xai-exp simple`, `xai-exp simple assumptions`, `xai-exp simple noise-prefix` |
| Random-subset mechanism audit | `paper-noise-random-subset.yaml`, `paper-noise-random-subset-v2.yaml` | `xai-exp simple noise-subset` |
| Random-order anchored control | `paper-noise-random-order-anchored.yaml` | `xai-exp simple noise-subset` |
| Relative robustness (null-anchored R_rel) | `paper-relative-robustness.yaml` | `xai-exp simple relative-robustness` |
| Matrix planning and compatibility archive | `full-matrix.yaml` | `xai-exp simple full-matrix` (requires `--confirm-full-matrix`) |

Each configuration declares an isolated storage namespace. To reproduce a
study on your own infrastructure, point `storage.remote_root` and the
`runtime` paths at your own storage and build the required assets (dataset
manifests, partitions, reference checkpoints, train-split means) with
`xai-exp phase0`; the guides below walk through each experiment end to end.

### Documentation

- [`docs/experiment-guide.md`](docs/experiment-guide.md) — the two-stage
  pipeline: scientific contract, artifact formats, configuration, and the
  command chains for the main experiment and every sub-experiment.
- [`docs/full-matrix.md`](docs/full-matrix.md) — matrix planning, provider
  gates, and the compatibility archive. It does not claim complete metric
  coverage.
- [`docs/noise-generalization.md`](docs/noise-generalization.md) — the
  PathMNIST / DenseNet-121 NOISE generalization cell.
- [`code/README.md`](code/README.md) — package installation, CLI overview,
  artifact schemas, storage modes, and testing.

## Building the reports

The conversion script reads the explicitly listed structured result roots and
regenerates every dataset report and the manifest:

```bash
python code/scripts/results_to_latex.py --print-counts
```

The repository does not require a local LaTeX installation. To build a PDF,
run your preferred LaTeX toolchain from a dataset directory, for example
`results/latex/by-dataset/imagenet/`.

## Relation to prior work

This repository supersedes our ESANN project
[*Theoretical Insights into Ensemble Strategies for Image Post-Hoc Explanation*](https://github.com/Xu-jinhua/Theoretical-Ensemble-Strategies-for-XAI).
The ESANN study introduced the ensemble strategies; the present work keeps its
locked explanation-method parameters and evaluates *when* rank aggregation of
explanations works. The released archive spans 14 datasets and preserves
historical dataset/model combinations, with formal metric tables and separate
compatibility diagnostics where a run stopped before Phase 2.

## License

The code and documentation in this repository are released under the
[MIT License](LICENSE). The vendored `Transformer-Explainability` snapshot
under `code/vendor/` is MIT licensed by its authors; see
[`code/vendor/README.md`](code/vendor/README.md).
