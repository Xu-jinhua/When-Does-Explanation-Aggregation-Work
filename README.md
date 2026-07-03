# When-Does-Explanation-Aggregation-Work

Official repository for the paper:

**When Does Aggregating Explanations Work?**

This repository contains the experimental results, table-generation files, plotting scripts, and configuration files associated with the paper. The study analyzes when aggregating multiple local post-hoc image explanations improves explanation quality, and when it fails.

## Overview

Post-hoc explanation methods for image classifiers often disagree with each other, especially when the input data, model architecture, or explanation method changes. A common response is to aggregate several explanations into a single explanation. However, aggregation is not automatically beneficial.

In the paper, attribution maps are converted into patch-level rankings, and these rankings are aggregated using rank-aggregation methods. The experiments are organized around three settings:

- **NAIVE:** directly aggregate all available explanations for the same image and model.
- **IND:** aggregate explanations generated from approximately independent explanatory sources.
- **NOISE:** aggregate only the subset of explanations retained by a distance-based rank-noise diagnostic.

The main empirical finding is that direct aggregation and approximate independence provide limited gains, whereas theory-guided subset selection under the distance-based rank-noise model substantially improves the performance of distance-based aggregation methods such as Borda Count and Kemeny-Young.

## Repository Structure

```text
.
├── configs/                 # Experiment configuration files
├── docs/                    # Additional documentation and notes
├── figures/
│   ├── naive/               # Figures for the NAIVE setting
│   ├── ind/                 # Figures for the IND setting
│   └── noise/               # Figures for the NOISE setting
├── results/
│   ├── raw/                 # Raw experimental logs and metric outputs
│   ├── processed/           # Processed results used by tables and figures
│   └── tables/              # Exported CSV/LaTeX tables
└── scripts/
    ├── figures/             # Scripts for generating figures
    └── tables/              # Scripts for generating tables
```

## Experimental Settings

The repository follows the three experimental settings reported in the paper.

### NAIVE

The NAIVE setting evaluates whether directly aggregating all available explanations is sufficient to outperform the best individual explanation method and Simple Averaging.

Expected outputs:

- representative single-case tables;
- top-$k$ and patch-size ablation figures;
- global win-rate summaries over datasets and predictive backbones.

### IND

The IND setting evaluates whether approximate independence among input explanations significantly affects aggregation performance. Explanations are generated from models trained on disjoint training subsets and compared against the corresponding NAIVE aggregation.

Expected outputs:

- NAIVE-vs-IND representative tables;
- top-$k$ and patch-size ablation figures;
- global summaries comparing IND against NAIVE and the best individual explanation.

### NOISE

The NOISE setting evaluates whether compatibility with a distance-based rank-noise model improves aggregation performance. This setting is used for aggregation methods with a distance-based interpretation, especially Borda Count and Kemeny-Young.

Expected outputs:

- representative NAIVE-vs-NOISE performance tables;
- Kolmogorov-Smirnov before/after subset-selection diagnostics;
- top-$k$ and patch-size ablation figures;
- global summaries for Borda(NOISE) and Kemeny(NOISE);
- subset-selection statistics.

## Results

The full experimental results will be released in `results/`.

Planned result files include:

- `results/raw/`: unprocessed metric logs;
- `results/processed/`: merged and cleaned result files;
- `results/tables/`: table-ready CSV or LaTeX files used in the paper.

## Figures

Figures are organized according to the three settings:

- `figures/naive/`
- `figures/ind/`
- `figures/noise/`

The corresponding plotting scripts will be placed in `scripts/figures/`.

## Citation

Citation information will be added after the paper metadata is finalized.

## Status

This repository is under preparation. Results, scripts, and documentation will be added as the experiments are finalized.
