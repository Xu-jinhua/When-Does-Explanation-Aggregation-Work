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

The experimental results are organized as dataset-level reports. Each report
collects the released model and aggregation-setting results for one dataset
and is distributed as a reader-friendly PDF together with its LaTeX source.

| Dataset | Models | PDF report | LaTeX source |
|:--|:--|:--|:--|
| ImageNet | ResNet-18, ViT-B/16 | [imagenet.pdf](results/pdf/by-dataset/imagenet.pdf) | [source](results/latex/by-dataset/imagenet/) |
| DermaMNIST | ResNet-18, ViT-B/16 | [dermamnist.pdf](results/pdf/by-dataset/dermamnist.pdf) | [source](results/latex/by-dataset/dermamnist/) |
| PathMNIST | DenseNet-121, ViT-B/16 | [pathmnist.pdf](results/pdf/by-dataset/pathmnist.pdf) | [source](results/latex/by-dataset/pathmnist/) |
| BloodMNIST | ViT-B/16 | [bloodmnist.pdf](results/pdf/by-dataset/bloodmnist.pdf) | [source](results/latex/by-dataset/bloodmnist/) |
| BreastMNIST | ViT-B/16 | [breastmnist.pdf](results/pdf/by-dataset/breastmnist.pdf) | [source](results/latex/by-dataset/breastmnist/) |
| OctMNIST | ViT-B/16 | [octmnist.pdf](results/pdf/by-dataset/octmnist.pdf) | [source](results/latex/by-dataset/octmnist/) |
| OrganCMNIST | ViT-B/16 | [organcmnist.pdf](results/pdf/by-dataset/organcmnist.pdf) | [source](results/latex/by-dataset/organcmnist/) |
| OrganSMNIST | ViT-B/16 | [organsmnist.pdf](results/pdf/by-dataset/organsmnist.pdf) | [source](results/latex/by-dataset/organsmnist/) |
| PneumoniaMNIST | ViT-B/16 | [pneumoniamnist.pdf](results/pdf/by-dataset/pneumoniamnist.pdf) | [source](results/latex/by-dataset/pneumoniamnist/) |
| RetinaMNIST | ViT-B/16 | [retinamnist.pdf](results/pdf/by-dataset/retinamnist.pdf) | [source](results/latex/by-dataset/retinamnist/) |

The reports contain 54 tables covering 10 datasets and 13 distinct
dataset-model combinations. The detailed model, protocol, and setting coverage
is provided in [`results/README.md`](results/README.md). The standalone
[paper NAIVE report](results/pdf/naive.pdf) also includes the paper ablations.

## Repository layout

```text
.
├── results/
│   ├── pdf/                 # Published result reports
│   ├── latex/               # LaTeX sources for the published PDFs
│   ├── processed/           # Processed result summaries
│   ├── raw/                 # Documented raw result exports
│   └── tables/              # Machine-readable and LaTeX table exports
├── code/                    # Experiment implementation
├── configs/                 # Experiment configurations
├── scripts/                 # Result and report-generation utilities
├── figures/                 # Figures organized by experimental setting
└── docs/                    # Supplementary documentation
```

## Building the reports

Run `latexmk` from a report's source directory. For example:

```bash
cd results/latex/naive
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```
