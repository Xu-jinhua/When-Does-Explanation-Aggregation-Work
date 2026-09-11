# Experiment guide: the two-stage pipeline

`xai-exp simple` is the paper-first two-stage pipeline. Phase 1 stores full
FP32 safetensors attributions; Phase 2 performs the paper-defined ranking,
aggregation, and masking evaluation. Validating or planning an experiment
never starts it, and the scheduler's short memory profiles only select or
measure runtime resources — they never process a dataset or produce an
explanation twice.

All commands below run from the `code/` directory of this repository with the
package installed (`pip install -e ".[gpu,dev]"`; see
[`../code/README.md`](../code/README.md)). Example invocations use the
shipped configurations under `code/configs/simple/`; substitute your own
config when reproducing on new infrastructure.

## Scientific contract

- Every model forward and every attribution is FP32. There is no BF16/FP32
  target split.
- A clean job explains that model's FP32 prediction. A perturbed job stores
  its current prediction but explains the matching clean FP32 prediction.
- Phase 1 preserves the current input's complete FP32 logits. Phase 2 uses
  their argmax as the unmasked metric prediction; it never substitutes the
  fixed clean explanation target for a perturbed input's current prediction.
- Phase 1 stores the complete signed attribution. It does not take an absolute
  value, reduce channels, make patches, or produce ranks.
- Phase 2 computes each method's patch score as
  `mean(abs(attribution))` over channels and pixels inside the patch, then
  makes a strict rank with deterministic row-major tie breaking.
- SimpleAvg first computes `mean(abs(attribution), channels)` for each method,
  spatially normalizes each method map (locked to `minmax` by default), takes
  the arithmetic mean across methods, and only then makes patch scores/ranks.
- The primary NAIVE, IND, and NOISE setting is `p=16, k=20`. The additional
  `p=8,14` outputs are only for declared patch-size ablations. FeatureAblation
  and Occlusion must use the same `p` as the corresponding Phase 2 evaluation.
- CNN and ViT use the method rosters in `configs/simple/methods.yaml`. The ViT
  roster contains eleven target-specific methods, including FeatureAblation,
  Occlusion, Chefer Transformer Attribution, original Partial/Full LRP,
  Gradient Attention Rollout, and Attention Grad-CAM. Guided Backprop,
  DeepLift, DeepLiftShap, raw Rollout, and last-layer raw attention are not ViT
  methods in this experiment.

The fixed ESANN settings are also in that method file: IG uses 50 steps and a
zero model-input baseline; DeepLift uses the normalized train-image mean;
GradientShap uses 20 samples; GradientShap and DeepLiftShap use the same fixed
three-member zero/Gaussian/train-mean distribution. The Gaussian is one
seeded `N(0,1)` tensor in model-input space and is independent of batch size.

## Artifacts

The Phase 1 unit is one dataset/model/split/condition/method variant. Its
`manifest.json` is published only after every shard has been uploaded and
verified. A shard is a safetensors file with:

```text
indices       int64   [N]
labels        int64   [N]
predictions   int64   [N]
logits        float32 [N, num_classes]
targets       int64   [N]
attributions  float32 [N, C, H, W]
```

`C` is normally 3 for input methods and 1 for attention methods. Both retain
the full input spatial size. The default shard contains 512 samples, but this
is independent of the explanation batch and Phase 2 inference batch. A batch
of 1024 therefore becomes two 512-sample shards.

Workers serialize each completed shard into a bounded RAM-backed spool, which
defaults to `/dev/shm/xai-simple/<experiment_id>`. A separate I/O thread then
publishes to the configured storage, verifies the result, writes an immutable
receipt, and only then removes the staged shard. Per-shard JSON records permit
resume; the final manifest is published only after every shard and receipt
succeeds.

The spool quota is shared across worker processes through SQLite. Unless YAML
overrides it, at most 64 GiB may be pending and at least 32 GiB of tmpfs
remains free. A dead process's reservation and private work directory are
reclaimed by the next writer. If publication is slower than generation, this
bound provides explicit backpressure instead of exhausting RAM.

Every Phase 1 shard and manifest uses schema version 2. The writer rejects a
shard unless `predictions == argmax(logits)` for every row, and Phase 2 checks
the same invariant again after reading. Schema-version checks prevent a
pre-logits shard from being mistaken for a resumable current artifact.

Each published manifest also records machine-readable `target_policy` and
`model_output_source` fields. Ordinary Phase 1 identifies the stored
condition logits and predictions as outputs of the task model. Assumptions
source Phase 1 instead identifies them as outputs of the complete
full-reference model and identifies the explanation target as that reference
model's clean FP32 prediction. Completion validation checks both fields and
their safetensors descriptions before reusing an immutable artifact.

## Storage configuration

`storage.remote_root` selects the artifact backend:

- **Local filesystem mode** — the first path segment contains no `:` (for
  example `./runs/quickstart/remote` or `/data/xai/experiments/main`).
  Payloads are published atomically, existing files are SHA-256-checked and
  never replaced, and the manifest is written last. Relative paths resolve
  against the configuration file's directory.
- **rclone remote mode** — the first segment contains a `:` (for example
  `myremote:experiments/main`). Publication uses `rclone copyto --immutable`
  with remote size and SHA-256 verification where the backend exposes it, and
  a streaming verification fallback otherwise. Workers never write large
  files through an rclone/FUSE mount.

`storage.rclone_binary` is optional and defaults to a `PATH` lookup;
`simple validate` checks that the binary exists even in local mode.

## Phase 2 data flow and metrics

Phase 2 loads one aligned 512-sample shard from every requested explanation
method. `indices`, `labels`, `predictions`, `logits`, and `targets` must match
exactly across methods. The full attribution remains signed on disk. For each
method, Phase 2 computes `mean(abs(attribution))` over channels and pixels in
each `p x p` patch, then turns descending patch scores into strict zero-based
ranks. Equal scores use the row-major patch index as the deterministic tie
break.

SimpleAvg follows the manuscript's separate path: take the attribution
absolute value, average channels, normalize each method's spatial map, average
the normalized maps across methods, average pixels within each patch, and
rank. Borda, RRF, Kemeny-Young, and Schulze consume the per-method rank
ballots. When `include_singles` is true, every source method's own rank is
evaluated by the same masking code as the ensemble rules.

After the NumPy FP32 attribution-to-patch and SimpleAvg map reductions have
fixed the ballots and SimpleAvg scores, the consensus stage uses a batched
Torch backend on the task's GPU. Borda and RRF rank complete shard tensors,
while Schulze batches pairwise winning-vote matrices and strongest paths. The
formal Kemeny approximation uses Borda as its single start and accepts the
globally best strict-improvement insertion move for at most 1024 moves per
sample. It optimizes the true Kendall disagreement objective, so its selected
result is checked to be no worse than Borda on every sample; this is not a
claim that its downstream Fidelity must exceed Borda. Matrix batches are
bounded by a workspace derived from the measured Phase 2 reservation, so
`p=8` is processed in smaller chunks without changing its result. Every shard
records move counts, cap hits, convergence, and Borda-to-final objective
improvement; the final manifest recomputes the corresponding split-wide totals
from all shard records.

For each rule and sample, the top `k` patches produce two raw-image inputs:

- `removed`: top-k pixels are replaced by the train-split mean; all other
  pixels remain from the input.
- `retained`: top-k pixels remain from the input; all other pixels are
  replaced by the train-split mean.

The model preprocessing is applied after masking. The Phase 1 prediction is
reused as the unmasked prediction, so Phase 2 forwards only `removed` and
`retained`. `inference_batch_size` is a hard bound on the actual model-forward
batch after those variants are concatenated; 512 therefore means at most 512
images in any one forward, not 1024 or 1536.

Let `y` be the true label, `f(x)` the reused Phase 1 prediction, `f(M(x))` the
removed prediction, and `f(Mbar(x))` the retained prediction. Per-sample
contributions are:

```text
F     = 1[f(x) == y] - 1[f(M(x)) == y]
Fbar  = 1[f(x) == y] - 1[f(Mbar(x)) == y]
C     = 1[f(M(x)) != f(x)]
Cbar  = 1[f(Mbar(x)) != f(x)]
```

The manifest reports each metric's mean over the full split. Each Phase 2
safetensors shard preserves row indices, labels, fixed explanation targets,
the reused unmasked predictions, every evaluated rank, and every removed and
retained prediction. These per-sample values are sufficient for later table,
bootstrap, or stratified post-processing without another model forward. For a
non-clean condition, the current manifest schema additionally reports the
legacy absolute metric difference and direction-aware signed degradation
relative to its aligned clean Phase 2 task.

### Signed robustness protocol revision

The paper-facing robustness decision is a signed quality change, rather than
the absolute difference exposed as `robustness.absolute`. Let superscript `o`
identify one perturbation condition, and let the unsuperscripted value be the
aligned clean result. The revised definitions are:

```text
R_F^o    = F^o - F
R_C^o    = C^o - C
R_Fbar^o = Fbar - Fbar^o
R_Cbar^o = Cbar - Cbar^o
```

All four revised `R` values are maximized. A positive value means that the
measured explanation quality improved under the perturbation, a negative value
means that it degraded, and zero means no change. The subtraction is reversed
for `Fbar` and `Cbar` because those two quality metrics are minimized, whereas
`F` and `C` are maximized. This definition uses the raw metrics directly; it
does not introduce an auxiliary transformation such as `1 - Fbar`.

Every reported `R` must be accompanied by the corresponding perturbed raw
quality value (`F^o`, `Fbar^o`, `C^o`, or `Cbar^o`) and its clean baseline.
For Best Individual, select one method separately for each clean quality
metric, using that metric's direction, and then keep that same method fixed
for all perturbation conditions of the corresponding `R`. An optional `|R|`
may be reported as change magnitude, but it must not be labeled as signed
robustness or degradation.

This is a protocol and post-processing revision, not a claim about the schema
of already published artifacts. Existing artifacts remain sufficient: each
perturbed manifest contains its raw quality metrics, its aligned clean
manifest contains the clean values, and the shards retain per-sample
sufficient statistics. The revised `R` therefore requires no new attribution
or masked-model inference, but all table summarizers and tracked CSV/JSON/TeX
results are generated under the revised result schema before they are used
with this definition. The script
`scripts/validate_signed_robustness_results.py` independently audits a signed
robustness summary from its raw quality values.

The signed convention follows the directional-performance principle used by
[Taori et al.](https://arxiv.org/abs/2007.00644) for effective and relative
robustness, and by
[Hendrycks and Dietterich](https://arxiv.org/abs/1903.12261) for corruption
degradation, while keeping the perturbed quality itself visible. Following
[Yeh et al.](https://arxiv.org/abs/1901.09392), it also keeps robustness
change separate from explanation sensitivity rather than treating a large
unsigned change as proof of degradation.

### Effective Robustness (ER)

ER is a separate, offline post-processing measure of perturbed explanation
quality after accounting for clean quality. For metric `m` and perturbation
condition `o`, it is:

```text
ER_m^o = S_m(Q^o) - beta_hat_m^o(S_m(Q))
```

`S` leaves `F` and `C` unchanged and negates `Fbar` and `Cbar`, so every ER
column is larger-is-better. `beta_hat` is a nonnegative-slope ordinary least
squares fit from clean to perturbed oriented quality, fitted separately for
each dataset/model/metric/condition cell. Its reference pool contains only
the original NAIVE single explainers for the same cell. It never contains an
aggregate, IND, NAIVE, or NOISE candidate. Direct single rows and
Best Individual use leave-one-out reference fits; aggregates use the full
single-explainer reference pool.

ER consumes completed summaries and the small original NAIVE Phase 2 manifests
only. It neither reads attribution shards nor reruns Phase 1, Phase 2, or a
model forward. For a completed compatible summary, run:

```bash
xai-exp simple effective-robustness \
  --config configs/simple/paper-main.yaml \
  --input-summary results/simple/paper-main-v1/signed-r-v2/table1/summary.json \
  --reference-summary results/simple/paper-main-v1/signed-r-v2/table1/summary.json \
  --manifest-source auto \
  --output-directory results/simple/paper-main-v1/effective-robustness-v1/table1
```

The output directory contains `er_table.csv` and `er_table_rows.tex`, each
with the 16 values `ER_{F,Fbar,C,Cbar}_{g,p,s,a}` needed by a future ER table.
`er_detail.csv` retains raw and oriented values, the expected perturbed value,
the curve identity, leave-one-out status, and extrapolation flag for every
endpoint. `reference_curves.csv` stores every regression coefficient and
diagnostic. The current adapter supports main NAIVE Table 1, IND, and the
completed dual-geometry and independent-geometry NOISE reports.
The p/k/fill/noise-strength ablations deliberately remain unsupported until
their own setting-matched single-explainer reference pools are published.

### NOISE ER significance

`effective-robustness-noise-bootstrap` compares the fixed selected NOISE-S and
NOISE-K q values with the exact matching q=11 NAIVE rule on the same images.
Each class-stratified paired image bootstrap replicate refits the original
single-explainer ER reference curve before calculating the NOISE-minus-NAIVE
gain. It reads Phase 2 prediction shards only and performs no model forward or
attribution. The test is conditional on complete-test-set q selection; it
quantifies image-sampling uncertainty, not held-out selector generalization.

```bash
xai-exp simple effective-robustness-noise-bootstrap \
  --config configs/simple/paper-main.yaml \
  --noise-prefix-config configs/simple/paper-noise-prefix-sweep.yaml \
  --naive-er-summary results/simple/paper-main-v1/effective-robustness-v1/table1/summary.json \
  --noise-er-summary results/simple/paper-noise-prefix-sweep-v1/effective-robustness-v1/independent-geometry/summary.json \
  --bootstrap-replicates 19999 \
  --output-directory results/simple/paper-noise-prefix-sweep-v1/effective-robustness-v1/noise-vs-naive-bootstrap
```

The report contains endpoint-level percentile intervals, one-sided positive
gain p values, and Holm correction across the complete endpoint family. For
the 640-endpoint main comparison at alpha 0.05, at least 12,799 bootstrap
repetitions are required before any Holm rejection is numerically attainable;
the CLI default is 19,999 and records this resolution check in `summary.json`.

## Configuration and commands

Copy `configs/simple/example.yaml`, replace its manifest, per-model
train-mean, and checkpoint placeholders, and give the run a new immutable
`experiment_id` and storage namespace. The train mean belongs to a
dataset-model pair because the deterministic resize/crop preprocessing can
differ between backbones. `scripts/train_example.sh` is a complete Phase 0
walk-through (manifest, reference partition, fine-tuned ResNet-18) that
produces exactly the assets a simple config binds.

Registered datasets (see `src/xai_ensemble/data/specs.py`: ImageNet100,
Food101, Places365, and the MedMNIST-224 family) may use `registry_key`. Any
other map-style dataset can instead declare `provider_factory:
module:function`, `provider_kwargs`, `image_column`, and `label_column`. The
provider receives `split`, `cache_directory`, and `keep_in_memory`; it returns
the row-addressable dataset. In either case, the manifest remains the source
of sample identity, labels, ordering, and provider row indices.

```bash
xai-exp simple validate --config configs/simple/my-experiment.yaml
xai-exp simple plan --config configs/simple/my-experiment.yaml \
  --output /path/to/plan.json
```

These commands are read-only except for the optional local plan JSON.
`validate` also verifies every configured dataset manifest, model-specific
mean, reference-checkpoint sidecar, and checkpoint SHA-256 before any GPU job
is submitted.

The one-time explanation profile identity is model architecture, final
explainer parameters, FP32, and input shape. The locked search grid is
`1,2,4,8,16,32,64,128,256,512,1024`; normal methods start at 256 and known
heavy methods start at their declared safe value. FeatureAblation/Occlusion
execute only a few sequential perturbations during profiling while preserving
the real batch/model/output memory footprint.

Phase 2 has a separate short profile keyed by model key, class count, FP32
input shape, and configured forward batch. It invokes the exact
removed/retained evaluator on synthetic raw images and records the CUDA peak.
It does not search a scientific parameter and does not touch Phase 1
artifacts. GPU aggregation runs before masking with a temporary workspace
capped below that measured reservation; its cache is released before model
inference.

The scheduler runs missing profiles first, at most one profile per otherwise
idle GPU so that each peak is attributable. It then keeps the selected Phase 1
batch and configured Phase 2 forward batch unchanged. Formal Phase 1 and
Phase 2 jobs are both placed by best fit using live free memory, each running
process's outstanding measured reservation, and the configured device
headroom (10% by default).

After a Phase 1 worker has computed and staged all missing shards, it drops
the model and CUDA tensors, empties the CUDA cache, and releases its GPU
reservation while its publication tail continues in the background. The
process remains `running` until every receipt and the final manifest are
complete, so a publication failure still fails the job and can never
masquerade as success.

```bash
# Profiles followed by Phase 1 only
xai-exp simple run --config configs/simple/my-experiment.yaml

# Profiles, Phase 1, then Phase 2
xai-exp simple run --config configs/simple/my-experiment.yaml --include-phase2

xai-exp simple status --config configs/simple/my-experiment.yaml

# State repair after a diagnosed failure; this does not run the queue.
xai-exp simple retry --config configs/simple/my-experiment.yaml \
  --job-id 'phase1:<exact-task-id>'
```

If a Phase 1 task has exhausted its retries, every Phase 2 task that depends
on that missing artifact remains `blocked`; all independent Phase 2 tasks
continue. The failed explainer is never silently removed from an ensemble,
because that would change the scientific method set.

SQLite stores commands, dependencies, attempts, PIDs, GPU placement, leases,
and log paths. Adversarial Phase 1 jobs depend on their one shared attacked
dataset as well as their explanation profile; all other Phase 1 dependencies
are unchanged. Phase 2 jobs depend on their source Phase 1 jobs, one reusable
inference profile, and, for a perturbed condition, the aligned clean Phase 2
result. A retry reruns the same immutable task/profile choice; verified shards
and final manifests make that retry idempotent.

Only job IDs generated by the currently loaded configuration are runnable.
Superseded rows may remain in the shared SQLite history, but the scheduler
does not select them or include them in terminal counts. Phase 2 artifact
roots also include the full task digest, so a changed aggregation policy
cannot collide with immutable shards from an earlier policy.

Phase 1 FeatureAblation and Occlusion jobs are GPU-exclusive: their short
profiles understate long-lived allocator growth across the three patch
variants, so exclusivity prevents same-card overcommit.

Without `--include-phase2`, the scheduler runs only explanation profiles,
shared adversarial-dataset producers, and Phase 1 tasks. This is a true
runtime scope: Phase 2 jobs already registered in the same SQLite database
remain pending or blocked and do not prevent the Phase 1-only run from
terminating.

Individual profile, Phase 1, or Phase 2 task IDs shown by `simple plan` may
be run directly with their corresponding subcommand. The scheduler is the
normal entry point. `simple retry` accepts only explicitly named failed jobs,
resets their attempt counters, and recursively reopens descendants blocked by
those jobs. It never starts a worker by itself.

## Custom perturbed conditions

A condition with `kind: factory` names `module:function`. The function is
called once per sample with keyword arguments `raw_images`, `labels`,
`indices`, `model`, `normalize`, and a deterministic row-derived `seed`, plus
the YAML `kwargs`. It must return one finite raw `[0,1]` image batch with the
same shape. Calling it per sample makes its result independent of all runtime
batch sizes and resume boundaries.

Exactly one clean condition is required. A non-clean Phase 2 job depends on
its matching clean job and reports both quality metrics and clean-versus-
condition robustness for the same rows and rules.

## Adversarial condition and feasibility pilot

`simple adversarial-pilot` is separate from the formal queue and records the
evidence used to lock the formal attack source and batch size. The formal
pipeline represents the attack as an upstream dataset task rather than
regenerating it inside an explainer.

The attack follows the implementation used in Sara Repetto's second study.
It works in raw `[0,1]` pixel space with `Linf <= 2/255`, fixes the clean FP32
prediction, minimizes the attacked source relevance at the clean source map's
stable top 10 percent, and adds `1e-4 * CE(f(x_adv), f(x))`. Adam uses 100
steps, learning rate `0.1`, and cosine decay. The original implementation's
batch-level best candidate is corrected here: every image independently keeps
its best prediction-preserving candidate, so batching is only a runtime
choice.

CNN attacks use DeepLift with a raw black-image baseline. ViT attacks use
Gradient Attention Rollout, implemented with a true second-order
attention-gradient graph. Raw Rollout and last-layer raw attention are not
valid attack sources because they provide no class-specific input gradient.
The attacked-dataset identity retains the legacy string
`TransformerAttribution`; manifests additionally record its scientific
semantics as `GradientAttentionRollout`. The resulting attacked image is
shared by every Phase 1 explainer; source-specific attacks per ensemble member
would define a different experiment.

The pilot searches real attack batches and selects the highest measured
sample-step throughput that remains within the configured GPU headroom. It
then runs the full optimizer on split-wide, evenly spaced samples. For
`p=8,14,16`, the report compares clean ranks against both the shared random
start and the optimized attack using changed-rank fraction, normalized
Spearman footrule, normalized Kendall distance, Spearman correlation, and
top-20 replacement. This control distinguishes a directed explanation attack
from the effect of merely adding bounded random noise.

```bash
xai-exp simple adversarial-pilot \
  --config configs/simple/paper-main.yaml \
  --model-id dermamnist-resnet18 \
  --batch-sizes 1,2,4,8,16,32,64,128,256 \
  --profile-steps 10 \
  --analysis-samples 32 \
  --steps 100 \
  --device cuda:0
```

Pilot outputs are written below
`<profile-directory>/../pilots/adversarial/<model-id>/<identity-prefix>/`.
`report.json` binds the dataset manifest, checkpoint, source, sample indices,
attack parameters, measurements, and rank diagnostics. The accompanying FP32
`attack-deltas.safetensors` stores row indices, labels, fixed targets, clean
and attacked logits, selected raw-pixel deltas, best steps, and source maps.
These pilot files are diagnostics and are never consumed as formal inputs.

The `adversarial-sara-2-255` condition expands to exactly one formal attack
task per dataset/model/split. Attack tasks are compute-exclusive: at most one
runs on a GPU. Each formal attacked dataset is saved below
`adversarial/<dataset>/<model>/<split>/<condition>/` in the configured
storage root. A 512-row safetensors shard contains `indices`, `labels`, fixed
clean `targets`, `clean_logits`, `adversarial_logits`, complete FP32
`adversarial_images`, FP32 `deltas`, and `best_steps`. Both the attacked image
and delta are retained so the exact experiment input is inspectable without
rerunning the optimizer. Payload and receipt are uploaded first, the shard
record second, and the immutable complete manifest last.

Partial verified shards are reused after interruption and a valid complete
manifest makes the task an immediate no-op. Every Phase 1 explainer and
Phase 2 loads the same shard by provider row index; each load checks labels,
clean targets, attacked logits, prediction preservation, raw pixel bounds, and
`Linf <= 2/255`. Phase 1 reuses the stored clean and attacked logits instead
of repeating those classifier forwards.

## Running the main experiment

`configs/simple/paper-main.yaml` is the four-cell ImageNet100/DermaMNIST by
ResNet-18/ViT-B16 configuration. It includes clean, Gaussian,
salt-and-pepper, speckle, and the saved Sara-style adversarial condition. It
expands to 33 explanation profiles, four Phase 2 inference profiles, four
adversarial datasets, 220 Phase 1 tasks, and 60 Phase 2 tasks: 321 scheduler
jobs in total.

Validate and launch the complete two-stage queue with:

```bash
xai-exp simple validate --config configs/simple/paper-main.yaml

xai-exp simple run --config configs/simple/paper-main.yaml --include-phase2
```

The same database may also be run without `--include-phase2` and extended
later, but this is an operational choice rather than a scientific requirement.
Training may use BF16 autocast; the saved weights, every explanation target,
every inference pass, and every attribution in the simple pipeline are FP32.

## Table 1 result summary

The Table 1 summarizer is CPU-only and reads the 20 completed Phase 2
manifests for the four dataset/model cells and five conditions. It fixes the
paper's main configuration to `p=16`, `k=20`, dataset-mean filling, Gaussian
0.15, salt-and-pepper 0.05, speckle 0.15, and the saved adversarial `2/255`
condition. Under the signed revision, Best Individual is anchored to the clean
quality metric and remains fixed across conditions.

```bash
xai-exp simple summarize \
  --config configs/simple/paper-main.yaml \
  --table table1 \
  --manifest-source local
```

`--table table1` enforces the paper's exact condition set (clean, Gaussian,
salt-and-pepper, speckle, adversarial). For any other condition set — for
example the clean-only `quickstart.yaml` — use `--table metrics`, which emits
a condition-agnostic `summary.json`, `metrics.csv`, and `metrics_rows.tex`
covering every completed Phase 2 task and every aggregation rule and single
method in it.

Each deterministic digest directory contains `summary.json`, the complete
`table1.csv`, table-ready `table1_rows.tex`, and `table1_analysis.csv`.
`summary.json` retains source task/manifests and the exact explainer selected
for every Oracle cell. The analysis CSV reports exact descriptive win and
pairwise better/equal/worse counts overall, by quality/noise metric group, and
by dataset/model. These counts are not significance tests; paired bootstrap or
confidence intervals require the per-sample Phase 2 sufficient-statistic
shards rather than only the final manifests.

Only table-level summaries and their provenance belong in the Git results
tree; raw explanations, per-sample Phase 2 shards, adversarial image banks,
checkpoints, logs, and scheduler state remain in the configured experiment
storage.

## Isolated NAIVE ablations

`configs/simple/paper-naive-ablations.yaml` is an additive, read-only consumer
of the completed main `p=16, k=20` artifacts. It has its own storage root,
scratch directory, tmpfs spool, logs, and SQLite database. It never submits a
main-experiment task and never writes below the main artifact root.

The accepted representative scope is ImageNet100 with the full-reference
ResNet18 on the test split. The plan covers:

- Table 2: `k=10,20,40,80`, reusing `k=20` and evaluating the three missing
  values for clean and the four center perturbations;
- Table 4: dataset-mean versus class-mean filling, reusing dataset-mean and
  evaluating class-mean for the same five conditions; and
- Table 5: Gaussian `0.10,0.15,0.20`, salt-and-pepper `0.03,0.05,0.08`,
  speckle `0.10,0.15,0.20`, and adversarial `1/255,2/255,4/255`.

The four center noise levels reuse immutable main ranks. The eight new edge
levels create 88 Phase 1 jobs, then eight reusable NAIVE rank banks, then
eight mask-game evaluations. Natural-noise levels of one type use the center
condition's `seed_group`, so each image sees one shared random field scaled or
thresholded at the requested severity. The two new adversarial datasets are
saved as complete 512-row safetensors shards before explanation generation.

Class means come only from the registered training artifact. For every test
image, the fill bank is indexed by the fixed clean explanation target saved in
Phase 1, not by the true label or a perturbed prediction. Evaluation reuses
the saved unmasked prediction and forwards only removed and retained variants.

The independent scheduler submits exactly 126 jobs: two adversarial dataset
producers, 88 Phase 1 jobs, eight GPU rank jobs, and 28 evaluations. Runtime
batch caps and memory reservations do not enter scientific task identities, so
they may be lowered after an OOM without discarding completed shards.

`validate` and `plan` do not submit jobs. `run` executes the complete
dependency graph and writes JSON/CSV summaries for Tables 2, 4, and 5 only
after every planned job succeeds.

```bash
xai-exp simple ablation validate --config configs/simple/paper-naive-ablations.yaml
xai-exp simple ablation plan     --config configs/simple/paper-naive-ablations.yaml
xai-exp simple ablation run      --config configs/simple/paper-naive-ablations.yaml
xai-exp simple ablation status   --config configs/simple/paper-naive-ablations.yaml
xai-exp simple ablation summarize --config configs/simple/paper-naive-ablations.yaml
```

After diagnosing a terminal failure, retry only its exact queue ID and restart
the same idempotent scheduler:

```bash
xai-exp simple ablation retry \
  --config configs/simple/paper-naive-ablations.yaml \
  --job-id '<kind>:<exact-task-id>'
```

The constructed rank-bank schema deliberately separates aggregation from the
mask-game parameters. Later IND and NOISE constructors publish the same
schema and reuse this evaluator without changing the NAIVE artifacts.

## Isolated IND, NAIVE, and NOISE experiment

`configs/simple/paper-assumptions.yaml` is a separate consumer and producer
namespace for the paper's IND and NOISE assumptions. It reuses the immutable
partition, checkpoint, and statistical objects already published by the base
run, but writes source explanations into a content-bound `source-rank-inputs/`
subtree. Its local scratch, tmpfs spool, log directory, and SQLite database
are isolated under their own namespace. It never adds jobs to `paper-main-v1`
and treats the current NAIVE artifacts as immutable read-only inputs.

The fixed scientific construction is:

- `p=16`, `k=20`, dataset-mean filling, and the complete test split;
- 11 disjoint stratified training partitions and 11 independently trained
  source models for each dataset/model cell;
- one deterministic bijection from the 11 source models to the 11 compatible
  explanation methods for IND;
- all 11 methods on each source model for the NAIVE comparator, with final metric
  values averaged across the 11 source models;
- the complete reference model's clean FP32 prediction as every source
  explanation target, its condition-specific FP32 predictions and logits
  as the stored unmasked outputs, and the complete reference model as the
  common mask-game evaluator;
- FeatureAblation and Occlusion generated only at their formal `p=16` variant;
  and
- NOISE ordered and selected on the complete test set, with
  `alpha=0.05`, 499 refitted bootstrap replicates, the largest non-rejected
  prefix, and the highest-p-value prefix as the all-rejected fallback.

Spearman/Borda and Kendall/Kemeny create separate selected collections. Each
selected collection is evaluated with SimpleAvg, Borda, RRF, Kemeny, and
Schulze. The NOISE Best Individual comparator is always selected from the
original complete NAIVE collection, never from the filtered collection.

The plan contains 837 jobs: four partition jobs, one reusable Spearman-Mallows
family job, 44 source-model training jobs, 220 source-model Phase 1 scope
jobs, eight NOISE selection jobs, 280 rank jobs, and 280 evaluation
jobs. Source Phase 1 still computes all 2,420 method artifacts, but each
published source shard retains only the five aligned reference fields, the
strict p=16 rank, and the normalized p=16 SimpleAvg patch score:

```text
indices                 int64   [N]
labels                  int64   [N]
predictions             int64   [N]
logits                  float32 [N, num_classes]
targets                 int64   [N]
rank__p016              int32   [N, 196]
simpleavg_score__p016   float32 [N, 196]
```

The full signed attribution and normalized 224x224 SimpleAvg spatial map exist
only transiently in CPU staging and are never published for assumptions source
models. The rank follows the paper's `mean(abs(attribution))` channel/patch
semantics. SimpleAvg normalizes each method before patch reduction and then
averages its saved patch scores across methods; this is algebraically the same
operation as averaging the normalized spatial maps before patch reduction.
The representation has a new immutable identity, and a numerical regression
checks score tolerance and resulting ranks on deterministic multi-method
fixtures.

Before creating the independent queue, `run` enforces a read-only base-input
gate. The gate requires all current main-run p=16 Phase 1 and Phase 2 tasks,
the four adversarial datasets, all needed explanation profiles, and all four
512-forward inference profiles. When the current main SQLite database exists,
its exact current-plan task states are authoritative and no redundant remote
scan is performed. If the database is unavailable, immutable manifests are
verified directly. `validate` reports both configuration validity and
`execution_ready`; the dedicated `readiness` command exits nonzero until every
base input is ready.

These first three commands are read-only and do not create assumptions jobs:

```bash
xai-exp simple assumptions validate  --config configs/simple/paper-assumptions.yaml
xai-exp simple assumptions readiness --config configs/simple/paper-assumptions.yaml
xai-exp simple assumptions plan      --config configs/simple/paper-assumptions.yaml
```

Only after `readiness` exits zero, start the independent DAG. There
is no `--include-phase2` switch because this command intentionally runs
training, Phase 1, selection, rank construction, and evaluation as one
resumable graph:

```bash
xai-exp simple assumptions run    --config configs/simple/paper-assumptions.yaml
xai-exp simple assumptions status --config configs/simple/paper-assumptions.yaml
```

### Paper-table priority IND path

The table-priority queue produces the `q=11` IND result while using
three source models per cell for its NAIVE comparator. The three
matched models are frozen from already complete artifacts without inspecting
metric values. Each IND source generates only its assigned explanation method.
Those method artifacts, IND rank artifacts, and IND evaluation artifacts keep
the exact identities used by the complete 11-by-11 queue, so the complete run
later verifies and reuses them.

The queue registers 308 jobs: four partitions, 44 training tasks, 220 assigned
source-method tasks, 20 IND rank tasks, and 20 IND evaluations. Existing
partitions, checkpoints, and method artifacts enter as succeeded after strict
validation. It refuses to start unless all 60 frozen NAIVE-comparator evaluations
are complete.

```bash
xai-exp simple assumptions table-priority-run \
  --config configs/simple/paper-assumptions.yaml

xai-exp simple assumptions table-priority-status \
  --config configs/simple/paper-assumptions.yaml
```

After terminal success, `table-priority-summarize` writes JSON, CSV, and TeX
rows. Best Individual is selected separately for each clean quality metric
after averaging that method over the three matched source models. The selected
method is then fixed for the corresponding robustness values. Summary schema
v2 derives signed `R` directly from the aligned clean and perturbed metrics
and does not use the legacy absolute field. JSON and CSV retain every
corresponding perturbed raw quality value. Matched rows also include the
sample standard deviation across the three source models for clean quality,
perturbed quality, and signed `R`.

```bash
xai-exp simple assumptions table-priority-summarize \
  --config configs/simple/paper-assumptions.yaml \
  --output-directory results/simple/paper-assumptions-compact-v1/ind-table-priority
```

After all jobs succeed, write the table-ready flat CSV and provenance-rich
JSON. The summary keeps IND, source-averaged NAIVE, both NOISE
distance models, original NAIVE rows, method selections, and every Oracle Best
Individual source method.

```bash
xai-exp simple assumptions summarize --config configs/simple/paper-assumptions.yaml
```

NOISE can be summarized independently after its eight selection and 40
evaluation tasks complete, without waiting for IND or NAIVE:

```bash
xai-exp simple assumptions summarize \
  --config configs/simple/paper-assumptions.yaml \
  --scope noise \
  --output-directory results/simple/paper-assumptions-compact-v1/noise
```

After diagnosing a terminal failure, retry only the exact queue ID and restart
the same idempotent scheduler:

```bash
xai-exp simple assumptions retry \
  --config configs/simple/paper-assumptions.yaml \
  --job-id '<kind>:<exact-task-id>'
```

## NOISE Fidelity-prefix sweep

`configs/simple/paper-noise-prefix-sweep.yaml` measures every Fidelity-ordered
q=2..11 NOISE prefix. It reads immutable paper-main p=16 artifacts and the
completed assumptions selection, and writes to a disjoint namespace; it never
regenerates Phase 1 attribution. Its subcommands follow the usual shape —
`validate`, `readiness` (`--refresh` re-checks base inputs), `plan`,
`evaluate --task-id`, `run`, `status`, `retry`, and `summarize
--output-directory`:

```bash
xai-exp simple noise-prefix validate  --config configs/simple/paper-noise-prefix-sweep.yaml
xai-exp simple noise-prefix readiness --config configs/simple/paper-noise-prefix-sweep.yaml
xai-exp simple noise-prefix run       --config configs/simple/paper-noise-prefix-sweep.yaml
xai-exp simple noise-prefix summarize \
  --config configs/simple/paper-noise-prefix-sweep.yaml \
  --output-directory results/simple/paper-noise-prefix-sweep-v1/q-sweep
```

## Noise-consistent random-subset mechanism audit

`configs/simple/paper-noise-random-subset-v2.yaml` is the active Phase-2-only
control for the concern that NOISE may improve only because it keeps the
highest-Fidelity methods. The initial bounded experiment uses
DermaMNIST/ResNet18, clean test images, `p=16`, `k=20`, and all five paper
aggregation rules. It reuses the immutable rank-ready inputs and the
independently frozen `q_S=2` and `q_K=3`; it never regenerates attribution.

For each geometry, the selector enumerates every non-reference subset at the
same q and uses a fixed-seed uniform draw from that complete candidate bank.
It builds each candidate's Borda or Kemeny center, fits the exact fixed-size
top-k subset-Mallows family, and computes the empirical-versus-analytic CDF
KS error. A candidate is eligible only when it is an interior fit whose KS
error is no greater than the frozen Fidelity-prefix reference at the same q.
Up to ten candidates are sampled uniformly without replacement from that
eligible pool. Neither candidate pool construction nor final sampling can
read aggregate F, Fbar, C, or Cbar. The task fails rather than weakening the
criterion when fewer than five eligible controls exist.

The resumable DAG has one rank-only selector followed by independent
Spearman/Borda and Kendall/Kemeny clean Phase-2 tasks. Every selected random
subset is evaluated; no result-dependent winner is chosen.

```bash
xai-exp simple noise-subset validate  --config configs/simple/paper-noise-random-subset-v2.yaml
xai-exp simple noise-subset readiness --config configs/simple/paper-noise-random-subset-v2.yaml
xai-exp simple noise-subset run       --config configs/simple/paper-noise-random-subset-v2.yaml
xai-exp simple noise-subset status    --config configs/simple/paper-noise-random-subset-v2.yaml
```

After terminal success, `summarize` writes candidate-level comparisons and
grouped win/tie/loss statistics against the exact corresponding
Fidelity-prefix at the same q:

```bash
xai-exp simple noise-subset summarize \
  --config configs/simple/paper-noise-random-subset-v2.yaml \
  --output-directory results/simple/paper-noise-random-subset-v2
```

`configs/simple/paper-noise-random-order-anchored.yaml` is the companion
random-order anchored control: it reuses the immutable p=16 clean rank bank,
changes only the method order, and applies the same anchored q selector as
NOISE. It is driven by the same `simple noise-subset` command group, and its
summary reports the anchored-control diagnostics.

## Relative robustness

`configs/simple/paper-relative-robustness.yaml` is a disjoint Phase-2-only
control namespace that reads existing rank-ready inputs, adversarial inputs,
and completed NAIVE/NOISE evaluations. It evaluates fixed random-mask controls
and calculates the null-anchored relative robustness
`R_rel = 1 - E_perturbed / E_clean`; lower is better, zero retains the clean
above-random quality, and a negative value improves it. A NOISE advantage
requires both a positive NAIVE-minus-NOISE R_rel gain and a positive
NOISE-minus-NAIVE perturbed excess gain.

```bash
xai-exp simple relative-robustness validate --config configs/simple/paper-relative-robustness.yaml
xai-exp simple relative-robustness plan     --config configs/simple/paper-relative-robustness.yaml
xai-exp simple relative-robustness run      --config configs/simple/paper-relative-robustness.yaml
xai-exp simple relative-robustness report \
  --config configs/simple/paper-relative-robustness.yaml \
  --output-directory results/simple/paper-relative-robustness-v1/report
```

The `report` step additionally accepts `--bootstrap-replicates`,
`--bootstrap-batch-size`, `--confidence`, and `--seed` for its interval
estimates.

## Reproducibility checklist

A completed run preserves all of the following:

- the versioned experiment configuration and its digest;
- pinned dataset manifests and split/content hashes;
- reference, IND, and matched-OVERLAP partition manifests;
- checkpoint hashes and training metadata;
- method locks, common-target traces, corruption seeds, and attack manifests;
- explanation/aggregation/evaluation artifact manifests and SHA-256 values;
- per-sample metric sufficient statistics and bootstrap seeds;
- the SQLite database, attempt logs, and the content-addressed manifest
  identities of the artifact store.

Without this set, a table can be numerically populated but is not considered a
reproducible formal result.
