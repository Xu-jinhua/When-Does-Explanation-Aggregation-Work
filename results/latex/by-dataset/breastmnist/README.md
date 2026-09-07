# BreastMNIST Results

This directory contains the LaTeX source for the dataset-level `BreastMNIST`
result report. The compiled report is
[`breastmnist.pdf`](../../../pdf/by-dataset/breastmnist.pdf).

| Protocol | Model | Settings |
|:--|:--|:--|
| Full-matrix protocol | ViT-B/16 | NAIVE, IND, NOISE-S, NOISE-K |

The tables use the released structured summaries from commit
[`d6e4b9bbfb3c`](https://github.com/Xu-jinhua/Workshop-When-Does-Explanations/tree/d6e4b9bbfb3c992ddd8eb7ec728e7b3cad85692f).

Build the report from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```
