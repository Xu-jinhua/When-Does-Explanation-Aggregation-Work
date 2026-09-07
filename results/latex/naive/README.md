# NAIVE report

`main.tex` compiles the published NAIVE result report. `Table_NAIVE.tex` is
the report-local table source adapted from the paper table so that the report
does not depend on the manuscript build or its review comments.

The report source was prepared from the Workshop manuscript snapshot at
`paper/Table_NAIVE.tex`. Numerical values and table-level experimental
metadata are retained in the table source.

Build from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```
