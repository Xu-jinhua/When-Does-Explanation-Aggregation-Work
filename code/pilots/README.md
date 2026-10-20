# Pre-experiment pilots

A pilot is a small, auditable experiment that fixes one free engineering or
statistical choice before the formal Phase 0--2 runs.  Each pilot writes a
JSON result with the common `PilotResult` schema (`name`, `status`,
`selected`, `measurements`, `failures`, `config_digest`, and a content
`digest`); `xai-exp pilot check` re-verifies the digest and refuses results
that did not pass.  Pilot definitions live in
[`../configs/pilots`](../configs/pilots); the implementations live in
[`../src/xai_ensemble/pilots`](../src/xai_ensemble/pilots).

| Pilot | Question it answers | Selected value |
|---|---|---|
| `ig_step_budget` | Which Integrated Gradients approximation depth (`n_steps`) gives the highest Fidelity on the validation split for this dataset-model pair? | `n_steps` per dataset-model pair |
| `baseline_selection` | Which reference (all-zero, dataset-mean, class-mean, Gaussian-noise, or uniform-noise tensor; sampled distributions for the SHAP-style methods) gives the highest validation Fidelity for each baseline-dependent method? | reference per method |
| `kemeny_solver` | Which Kemeny-Young local-search budget reproduces the exact optimum on small synthetic instances, never does worse than its Borda start, and stays within its time limit? | `n_starts`, `neighborhood`, `max_passes`, `time_limit_seconds` |
| `noise_gof_cost` | Is the calibrated Spearman distance family stable, and which refitted parametric-bootstrap replicate budget `B` is affordable and precise enough? | `bootstrap_replicates` plus the family artifact binding |

## Running a pilot

Run the commands from `code/` with `PYTHONPATH=src`.  The two GPU pilots
evaluate on an evenly spaced subset of the configured validation split with
the locked evaluation geometry (`p=16`, `k=20`, dataset-mean fill):

```bash
python -m xai_ensemble.cli pilot ig-step-budget \
  --config configs/pilots/ig_step_budget.yaml \
  --experiment configs/simple/paper-main.yaml \
  --model-id dermamnist-resnet18 \
  --output runs/pilots/ig-step-budget--dermamnist-resnet18.json

python -m xai_ensemble.cli pilot baseline-selection \
  --config configs/pilots/baseline_selection.yaml \
  --experiment configs/simple/paper-main.yaml \
  --model-id dermamnist-resnet18 \
  --output runs/pilots/baseline-selection--dermamnist-resnet18.json
```

The two CPU pilots need no dataset or GPU; the Kemeny benchmark compares the
solver grid against an exact subset-DP oracle on synthetic ballots, and the
NOISE cost pilot calibrates the Spearman base measure on synthetic validation
distances before adjudicating the bootstrap budget:

```bash
python -m xai_ensemble.cli pilot kemeny-solver \
  --config configs/pilots/kemeny_solver.yaml \
  --output runs/pilots/kemeny-solver.json

python -m xai_ensemble.cli pilot noise-gof-cost \
  --config configs/pilots/noise_gof_cost.yaml \
  --n-items 196 \
  --artifact runs/pilots/spearman-family.npz \
  --output runs/pilots/noise-gof-cost.json
```

Verify any produced result with:

```bash
python -m xai_ensemble.cli pilot check runs/pilots/kemeny-solver.json
```

## How the selected values are pinned

A pilot result is data, not configuration: formal runs never read pilot
JSON files.  The selected values are transferred into the versioned
configuration that formal runs consume, and the pilot result documents why
those values were chosen:

- the Integrated Gradients `n_steps`, the DeepLift `baseline`, and the
  GradientShap/DeepLiftShap `baseline_distribution` and `n_samples` live in
  [`../configs/simple/methods.yaml`](../configs/simple/methods.yaml), the
  single source of formal explainer parameters;
- the Kemeny-Young budget (`kemeny_starts`, `kemeny_max_passes`) lives in the
  `phase2:` block of each experiment configuration;
- the NOISE bootstrap budget is adjudicated against
  [`../configs/pilots/noise_gof_cost.yaml`](../configs/pilots/noise_gof_cost.yaml):
  its `spearman_mallows_approximation` section fixes the calibration
  algorithm, resource limits, and stability gates, the passing result records
  the selected `bootstrap_replicates` together with the calibrated Spearman
  family artifact (`path` plus `sha256`), and each assumptions experiment
  binds that artifact through its `spearman_artifact` key and loads it with
  `xai_ensemble.phase2.selection:locked_spearman_family_factory`.

A pilot whose status is `failed` is a blocker: preserve the result, fix the
cause (more samples, a different candidate set, or a reviewed configuration
change), and rerun rather than weakening the pass criteria after seeing the
outcome.
