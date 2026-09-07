# OrganSMNIST Results

This directory contains the LaTeX source for the dataset-level `OrganSMNIST`
result report. The compiled report is
[`organsmnist.pdf`](../../../pdf/by-dataset/organsmnist.pdf).

| Protocol | Model | Settings |
|:--|:--|:--|
| Full-matrix protocol | ViT-B/16 | NAIVE, NOISE-S, NOISE-K |

The tables use the released structured summaries from commit
[`d6e4b9bbfb3c`](https://github.com/Xu-jinhua/Workshop-When-Does-Explanations/tree/d6e4b9bbfb3c992ddd8eb7ec728e7b3cad85692f).

Build the report from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```
