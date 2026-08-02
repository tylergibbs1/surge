# Modal deployment

Modal hosts the FastAPI inference service, EIA load ingestion, and observed
ASOS weather ingestion. This is an operator runbook, not an availability or
cost promise. The full restore and rollback procedure is in
[`docs/operations.md`](../docs/operations.md).

## Current topology

The checked-in v0.2 configuration uses:

- Python 3.12 and a pinned Hugging Face model revision;
- app name `surge-api-v02` and volume `surge-grid-v02`, both configurable when
  `modal_app/app.py` is loaded;
- hourly EIA-930 refresh at `:05` UTC for the 53 demand-reporting BAs;
- six-hourly ASOS refresh at `:35` UTC for PJM, CISO, ERCO, MISO, NYIS, ISNE,
  and SWPP;
- hourly immutable outcome settlement at `:45` UTC after the final forecast
  point has matured by 72 hours;
- one warm API container, up to 20 burst containers, and a 600-second
  scaledown window;
- a 60-second volume-reload TTL in warm API containers; and
- a distinct `ledger_publisher_app`, capped at one container, that reloads the
  volume before a seven-RTO batch and commits it before returning success.

The seven RTO/ISOs are the v0.2 publication and verification surface. Their
Python ledger records stage independently, but public run listings and the
scoreboard remain hidden until an immutable complete-run marker covers all
seven compatible issuances. Direct issuance detail stays available for audit.
The downstream Vercel `current` pointer is a separate atomic boundary. The
wider API remains best-effort. `load-v2-core` uses observed temperature in the
historical context but has calendar-only future covariates; the ASOS job is not
a forecast-weather feed.

A clean checkout is deployable without the ignored `data_snapshot/` directory.
In that case the image contains an empty seed and readiness remains false until
`bootstrap_data` establishes at least 2,048 usable load and weather rows for
each RTO and completes a real 24-hour inference probe for all seven. A verified
local `data_snapshot/` can still be baked in as a seed.

## Prerequisites and configuration

Install and authenticate the Modal CLI, then create both named secrets:

```bash
python -m pip install modal
modal token new
modal secret create eia-api EIA_API_KEY='<value>'
modal secret create surge-ledger SURGE_LEDGER_KEY='<random-value>'
```

Keep values out of shell history where your environment supports prompted or
file-based secret input. The names `eia-api` and `surge-ledger` are part of the
checked-in deployment definition.

Deploy from a reviewed tag or commit and pin the deployment identity before
loading the Modal module:

```bash
export SURGE_MODAL_APP=surge-api-v02
export SURGE_MODAL_VOLUME=surge-grid-v02
export SURGE_MODEL_PATH=amazon/chronos-2
export SURGE_MODEL_REVISION=29ec3766d36d6f73f0696f85560a422f50e8498c
export SURGE_CODE_REVISION="$(git rev-parse HEAD)"
```

A custom adapter must additionally set `SURGE_MODEL_FEATURE_SPEC_SHA256` to
the training manifest's `load-v2-core` digest and
`SURGE_MODEL_ARTIFACT_SHA256` to the deterministic hash of the extracted model
tree. Compute the latter only after extraction and verification:

```bash
python -c 'from surge.model_loader import artifact_sha256; print(artifact_sha256("/absolute/path/to/model"))'
```

An archive checksum proves transfer integrity; it is not the extracted model
identity expected by the API. The API refuses authenticated ledger commits
when either attestation is missing or mismatched.

Use a new versioned app and volume name for a restore rehearsal. Do not point a
candidate release at the prior production volume.

## Establish data

For a network bootstrap into the selected volume:

```bash
modal run modal_app/app.py::bootstrap_data --days 120
```

The function fetches load and observed weather for the seven trusted RTOs,
commits the volume, checks per-RTO recency and usable history, loads the pinned
model, and completes a real 24-hour forecast for every RTO. It exits non-zero
on partial success. Its JSON log is part of the restore evidence.

Alternatively, build a verified image seed from a validated local store before
loading or deploying `modal_app/app.py`:

```bash
export SURGE_DATA_DIR=/absolute/path/to/validated/surge-data
python scripts/rebuild_data_snapshot.py \
  --output data_snapshot \
  --max-input-age-hours 12 \
  --force
python scripts/rebuild_data_snapshot.py --verify data_snapshot
```

Do not seed from an unverified personal data directory. The bootstrap is still
useful after seeding because it refreshes the current 120-day window and applies
the same postcondition checks. `--force` replaces only an existing directory
that already carries the snapshot manifest marker; it will not repurpose an
arbitrary `data_snapshot/` directory.

## Deploy and verify

```bash
git status --short
git rev-parse HEAD
modal deploy modal_app/app.py
```

Modal prints two web endpoints. The burst-scaled `fastapi_app` is read-only for
ledger purposes; `ledger_publisher_app` is the only production write origin.
Probe liveness, readiness, metadata, and every trusted RTO:

```bash
export SURGE_API_URL='https://<account>--surge-api-v02-fastapi-app.modal.run'
export SURGE_PUBLISHER_URL='https://<account>--surge-api-v02-ledger-publisher-app.modal.run'
curl --fail "$SURGE_API_URL/live"
curl --fail "$SURGE_API_URL/ready"
curl --fail "$SURGE_API_URL/health"
curl --fail "$SURGE_API_URL/bas"
for ba in PJM CISO ERCO MISO NYIS ISNE SWPP; do
  curl --fail "$SURGE_API_URL/forecast/$ba?horizon=24"
done
```

HTTP success is insufficient. Confirm the exact model and code revisions,
feature specification, input cutoff, 24 consecutive future hours, finite
values, and ordered p10/p50/p90. Health checks global load and weather maxima;
it does not prove per-RTO freshness. Forecast construction rejects the requested
RTO when its load or observed-weather watermark is more than 12 hours old, so
the seven forecast probes are a mandatory end-to-end postcondition.

## Wire a Vercel preview

Set only server-side variables in the preview project:

```text
SURGE_API_URL=https://<account>--surge-api-v02-fastapi-app.modal.run
SURGE_PUBLISHER_URL=https://<account>--surge-api-v02-ledger-publisher-app.modal.run
SURGE_ALLOWED_API_HOSTS=<account>--surge-api-v02-ledger-publisher-app.modal.run
BLOB_READ_WRITE_TOKEN=<managed by Vercel Blob>
BAKE_SECRET=<random shared secret>
SURGE_LEDGER_KEY=<same value stored in the Modal surge-ledger secret>
```

Run a complete authenticated bake and read-side smoke test in preview before
promotion. The bake makes one `POST /ledger/runs/bake` request to the serialized
publisher and requires its `X-Surge-Volume-Committed: true` attestation.
`SURGE_ALLOWED_API_HOSTS` is an exact comma-separated publisher allowlist; add a
versioned restore hostname before switching the publisher URL. A partial bake
must leave the prior complete run current.

## Scaling, restore, and rollback

At v0.2 the API uses `cpu=2`, `memory=4096`, `min_containers=1`, and
`max_containers=20`. One warm container means the deployment does not scale to
zero. Review current provider pricing and measured usage directly.

Safe recovery uses a new `SURGE_MODAL_VOLUME`, validates it, deploys a preview,
and preserves the previous app, model revision, and volume for rollback. The
code supports this path and no longer requires a repository snapshot, but a
hosted end-to-end restore rehearsal has not yet been executed. Do not describe
the operational recovery path as verified until that evidence is recorded.
