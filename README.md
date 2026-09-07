# When Does Aggregating Explanations Work?

## A Rank-Aggregation Analysis of Patch-Based Saliency Maps

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

The experimental results are organized by the three settings used in the
paper. Each report is distributed as a reader-friendly PDF together with the
LaTeX source used to produce it.

| Setting | PDF report | LaTeX source |
|:--|:--|:--|
| NAIVE | [naive.pdf](results/pdf/naive.pdf) | [source](results/latex/naive/) |
| IND | [ind.pdf](results/pdf/ind.pdf) | [source](results/latex/ind/) |
| NOISE | [noise.pdf](results/pdf/noise.pdf) | [source](results/latex/noise/) |

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
