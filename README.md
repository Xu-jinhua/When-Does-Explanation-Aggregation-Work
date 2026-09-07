# When Does Explanation Aggregation Work?

Official result repository for the paper *When Does Explanation Aggregation
Work?*

The repository is being released in stages. The first stage publishes
reader-friendly result reports as compiled PDF files together with the LaTeX
sources used to produce them. The experiment implementation will be added to
the reserved `code/`, `configs/`, and `scripts/` locations later.

## Repository layout

```text
.
├── results/
│   ├── pdf/                 # Published result reports
│   ├── latex/               # LaTeX sources for the published PDFs
│   ├── processed/           # Future machine-readable summaries
│   ├── raw/                 # Reserved for documented raw exports
│   └── tables/              # Future CSV/LaTeX table exports
├── code/                    # Reserved for the reproducible implementation
├── configs/                 # Reserved for experiment configurations
├── scripts/                 # Reserved for table and figure-generation tools
├── figures/                 # Reserved for released figures
└── docs/                    # Release notes and supplementary documentation
```

## Published reports

Reports are grouped by the experimental setting used in the paper:

- `results/pdf/naive.pdf`
- `results/pdf/ind.pdf`
- `results/pdf/noise.pdf`

Each report will have a matching source directory under `results/latex/`.
The tables retain their experimental parameters, metric directions, and
dataset/model labels in the table headers and captions.

## Release status

The repository structure is established first. Result PDFs and their LaTeX
sources will be added one report at a time. Code and configuration files are
reserved for a later release.

## Citation

Citation information will be added when the paper metadata is finalized.
