# Reproducibility guide

Reproducibility means more than rerunning a Python command. A Surge Grid result
is reproducible only when code, model, configuration, input vintages, origins,
and output hashes are preserved together.

## Capture the environment

Start from a clean tagged checkout:

```bash
git status --short
git rev-parse HEAD
python --version
python -m pip --version
python -m pip freeze --all > environment.txt
nvidia-smi > accelerator.txt 2>&1 || true
```

Record the OS and accelerator even for CPU inference because numerical kernels
and dependency resolution can change results. Do not put credentials or home
directory paths in the published bundle.

## Verify the data snapshot

```bash
shasum -a 256 -c surge-data-v0.2.0.tar.zst.sha256
mkdir -p reproduced-data
zstd -dc surge-data-v0.2.0.tar.zst | tar -xf - -C reproduced-data
python scripts/rebuild_data_snapshot.py --verify reproduced-data
export SURGE_DATA_DIR="$PWD/reproduced-data"
```

The snapshot manifest must name source retrieval times, row counts, coverage,
and hashes. The operational recovery snapshot compacts to current canonical
rows and is therefore insufficient for historical replay. Replay and
live-forward result bundles must separately preserve revision availability and
immutable issuance records.

## Run the archived oracle code path

The current v0.2 feature contract deliberately cannot run the old oracle
feature path. Replaying that implementation requires a detached checkout of
the archived source and the exact historical input snapshot:

```bash
git worktree add --detach ../surge-v001-oracle 36ceaff
cd ../surge-v001-oracle
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[api]" huggingface_hub
git rev-parse HEAD > code-revision.txt
python -m pip freeze --all > environment.txt
nvidia-smi > accelerator.txt 2>&1 || true

export SURGE_DATA_DIR=/absolute/path/to/exact-archived-snapshot
hf download Tylerbry1/surge-fm-v3 \
  --revision b84726ca520b9d443236d025a000cc95616a334c \
  --local-dir surge-fm-v3-b84726c
python -m experiments.run_c2 v3-oracle-test \
  '{"base":"./surge-fm-v3-b84726c","context":2048,"horizon":24,"on":"test","batch_size":16,"bootstrap":1000,"seed":42,"with_gen":true}' \
  | tee v3-oracle-test.log
```

Run this on a compatible CUDA host. This legacy model download is only for the
explicitly labeled historical reproduction; never use it for a v0.2 restore or
deployment. The archived loader defaults to all demand-reporting BAs.
`with_gen=true` allows realized future wind/solar where
available; temperature is also realized. Set `with_gen=false` only for a
separately named temperature-only oracle configuration.

Hash outputs immediately:

```bash
shasum -a 256 \
  code-revision.txt environment.txt accelerator.txt v3-oracle-test.log \
  > SHA256SUMS
```

The exact original v3 input snapshot and configuration hash were not released,
so the command documents the archived code path but cannot by itself reproduce
the README value byte-for-byte. Do not substitute a current canonical snapshot
and describe the output as a reproduction. A v0.2 claim remains provisional
until its complete result bundle satisfies
[benchmark-protocol.md](benchmark-protocol.md).

Current `experiments.run_c2` uses calendar-only future covariates and scores
p50, but its default `retrospective_final` source selection may include later
revisions. It is therefore a non-oracle development run, not a lane-2 vintage
replay, unless the complete availability-time requirements are independently
satisfied.

## Replay and live-forward runs

Vintage replay requires an archived issue-time weather dataset and source
availability timestamps. Live-forward evaluation requires immutable issuance
records that predate outcomes. Neither lane may fall back to the oracle feature
builder. Use the leakage gates and result-bundle layout in the
[benchmark protocol](benchmark-protocol.md).

## Validation-only interval calibration

From a source checkout with the training dependencies installed, run the
rolling conformal harness against a checksummed development snapshot containing
the complete 2024 validation window:

```bash
python -m pip install -e ".[train]"
export SURGE_DATA_DIR=/absolute/path/to/checksummed-2024-development-snapshot
export SURGE_DATA_SNAPSHOT_SHA256='<snapshot-manifest-sha256>'
python -m experiments.run_conformal_c2 \
  --model amazon/chronos-2 \
  --model-revision 29ec3766d36d6f73f0696f85560a422f50e8498c \
  --code-revision "$(git rev-parse HEAD)" \
  --data-snapshot-sha256 "$SURGE_DATA_SNAPSHOT_SHA256" \
  --out artifacts/conformal-validation.json
```

The current runner requires a compatible CUDA host. It opens only 2024
validation origins, tries the declared per-BA and
seven-RTO pooled windows, and records every candidate plus the selection rule.
The compact recovery snapshot is usable here only if its manifest proves that
full window exists; it is still not an availability-vintage replay dataset.
Window and pooling choices must be frozen before any future locked test that
proposes to apply them. The v0.2 H100 `run_c2` test described below evaluates
raw model quantiles and does not consume this calibration artifact. Its
`retrospective_final` availability mode is an explicit limitation: this output
is not vintage replay or live-forward calibration evidence, and no public
calibration claim should be made until the frozen method succeeds
prospectively with maturity-aware outcomes.

## Auditable fine-tune selection

Use a new empty output directory and immutable identities. The v0.2 trainer
accepts exactly the seven trust RTOs:

```bash
export SURGE_DATA_DIR=/absolute/path/to/checksummed-development-snapshot
export SURGE_DATA_SNAPSHOT_SHA256='<64-character-manifest-sha256>'
python -m experiments.finetune_c2 \
  --base amazon/chronos-2 \
  --base-model-id amazon/chronos-2 \
  --base-revision 29ec3766d36d6f73f0696f85560a422f50e8498c \
  --bas PJM CISO ERCO MISO NYIS ISNE SWPP \
  --code-revision "$(git rev-parse HEAD)" \
  --data-snapshot-sha256 "$SURGE_DATA_SNAPSHOT_SHA256" \
  --diagnostic-origins 90 \
  --out artifacts/candidate-v0.2
```

The command requires a compatible CUDA environment and atomically reserves a
new output path, refusing even a pre-existing empty directory. It queries no
valid time on or after 2025-01-01. It scores the returned best-loaded candidate
on train and 2024 validation tails, then scores the pinned upstream baseline on
the same 2024 origins. Checkpoint loss uses an earlier 90-origin validation
cohort; the latest 90 origins are reserved for promotion. Every origin is
shared by all seven RTOs and has 24 finite labels, selected by target
availability before any prediction. Each result records skipped/reserved counts
and exact origin hashes. Remote model IDs are loaded with `--base-revision`; local artifact
paths, including historical v3 reproductions, are loaded without forwarding a
remote revision while the public upstream identity remains recorded.
The promotion command itself accepts only the pinned
`amazon/chronos-2@29ec3766d36d6f73f0696f85560a422f50e8498c` lineage. Run legacy
v3 or custom checkpoints through non-test `experiments.run_c2` ablation lanes;
they are deliberately ineligible for `best/` and `surge-promotion.json`.

Always retain `surge-overfit-audit.json` and
`surge-training-manifest.json`. The audit includes the thresholds, gate-by-gate
decision, per-RTO metrics, worst-RTO/dispersion measures, baseline comparison,
training trace, dependency versions, and `test_opened=false`. A rejected run
exits non-zero and contains only `candidate-unpromoted/`; it has no `best/` and
no promotion marker. An eligible run additionally contains `best/` and
`surge-promotion.json`. Downstream test or release tooling must require and
hash-verify that marker.

For the frozen H100 experiment, run the declared 1,000- and 2,000-step jobs in
sibling directories and freeze the winner with the precommitted selector:

```bash
python -m experiments.select_c2_candidate \
  --candidate artifacts/official-lora-1000 \
  --candidate artifacts/official-lora-2000 \
  --out artifacts/v0.2-h100-selection.json
```

The selector admits only the pinned release-safe upstream lineage, verifies
both candidate chains, applies the composite MASE/scaled-WIS rule from
`docs/model-selection-experiment.md`, and writes the result without replacing
an existing selection artifact.

The current runner refuses to open the test partition without that complete
promotion chain:

```bash
python -m experiments.select_c2_candidate \
  --candidate artifacts/official-lora-1000 \
  --candidate artifacts/official-lora-2000 \
  --out artifacts/v0.2-h100-selection.json
export SURGE_LOCKED_TEST_REGISTRY=/absolute/path/to/operator-controlled-registry
python -m experiments.run_c2 locked-once-v0.2 \
  '{"base":"artifacts/official-lora-1000/best","bas":["PJM","CISO","ERCO","MISO","NYIS","ISNE","SWPP"],"context":2048,"horizon":24,"on":"test","test_protocol":"locked-once","selection_artifact":"artifacts/v0.2-h100-selection.json","promotion_artifact":"artifacts/official-lora-1000/surge-promotion.json","base_revision":"29ec3766d36d6f73f0696f85560a422f50e8498c","code_revision":"<same-reviewed-commit-sha-as-manifest>","data_snapshot_sha256":"77d80d4031e2391808103ef29bb182b3ee2469cec1c24ae00569d217bd48a4c0","bootstrap":2000,"seed":42,"per_step":true}'
```

Use the `base` and promotion path named by the selector's `winner`; the sample
shows the 1,000-step path only as an example. The verifier checks the manifest
and audit hashes, eligible policy
decision, unopened-test flags, every file size and SHA-256 below `best/`, and
that `base` resolves to that promoted directory. The base, code, data snapshot,
feature contract, and ordered seven-RTO identity must match the training
manifest. It also verifies the clean executing Git revision, the full frozen
snapshot byte inventory, exact training dependency versions, and the captured
Python/platform/CUDA/cuDNN/H100/determinism/TF32 runtime identity. Before
loading any 2025 row, the runner exclusively reserves the outcome-independent
frozen experiment-protocol hash in
`SURGE_LOCKED_TEST_REGISTRY` and creates `surge-locked-test-receipt.json` beside
the selection artifact. Completion stores the metric payload and its hash in
both records; a crash leaves a consuming `started` reservation, and any second
attempt fails closed. The immutable result retains full-precision metrics;
rounding is applied only to terminal display. The promoted context,
horizon, and generation setting are mandatory, with `step=24` and no origin
cap, 2,000 paired origin-block resamples, seed 42, and per-horizon metrics, so
the test window cannot be adjusted after inspection.

These diagnostics use `retrospective_final` data and fixed governance
thresholds. They can reject obvious generalization failures but cannot prove
future robustness. Chronos chooses its returned checkpoint from aggregate loss
across the frozen rolling validation tasks; the post-fit MASE/WIS audit does not
recover and compare every discarded checkpoint.

## Reproduce the local API environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "surge-grid[api]==0.2.0"
export SURGE_DATA_DIR="$PWD/reproduced-data"
export SURGE_MODEL_PATH="$PWD/chronos-2-29ec376"
export SURGE_MODEL_REVISION=29ec3766d36d6f73f0696f85560a422f50e8498c
export SURGE_MODEL_NAME=chronos-2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uvicorn surge.api.main:app --host 127.0.0.1 --port 8000
```

Save one representative response with its headers and hash it. Confirm its
model and data revisions before comparing values across machines.
