# Modal deployment

Host the Surge forecast API on [Modal](https://modal.com) — Python-native
serverless, builds on their Linux infra (no local Docker), scale-to-zero.

## One-time setup

```bash
pip install modal
modal token new        # browser flow creates ~/.modal.toml
```

## Deploy

```bash
modal deploy modal_app/app.py
```

First deploy builds the image remotely (~3–5 min, includes pre-pulling the
`Tylerbry1/surge-fm-v2` checkpoint). Later deploys reuse the layer cache
and take ~15 s.

After deploy you'll see:

```
✓ Created objects.
App deployed to https://modal.com/apps/<account>/surge-api
└─ surge-api-fastapi_app  https://<account>--surge-api-fastapi-app.modal.run
```

Test:

```bash
curl 'https://<account>--surge-api-fastapi-app.modal.run/forecast/PJM?horizon=24'
```

## Wire the playground

In `web/playground/.env.local`:

```
SURGE_API_URL=https://<account>--surge-api-fastapi-app.modal.run
```

No `RUNPOD_API_KEY` needed — Modal endpoints are publicly reachable by
default. If you want auth, add a `modal.fastapi_endpoint` token or put
`@modal.Secret.from_name("api-key")` in front of the app.

## Cost

| Tier | Compute | This load |
|---|---|---|
| Free | $30/month credits | covers ≥ 250k requests/month |
| Paid | $0.000038/CPU-sec + $0.00000036/RAM-MB-sec | ~$0.05 per 1k requests |

Surge's expected launch traffic (~1k req/day) fits comfortably in the free
tier indefinitely.

## Scaling knobs

Edit `modal_app/app.py`:

- `cpu=2.0` — CPU cores per container. `1.0` is enough but 2.0 halves the
  inference time.
- `memory=4096` — MB per container.
- `max_containers=4` — caps horizontal scale. Bump for burst traffic.
- `scaledown_window=300` — seconds of idle before a container is reclaimed.
  300 s = one warm container covers ~30k requests/month at 1/s peak.
- `min_containers=0` — scale to zero when idle (default). Set `1` to keep
  one warm, adding ~$5/month but killing cold-start latency.

## Updating the data snapshot

The parquet store ships *inside* the image via `add_local_dir`. To refresh:

```bash
# locally
python -m surge.ingest --days 30
python scripts/rebuild_data_snapshot.py   # regenerates /data_snapshot

# redeploy
modal deploy modal_app/app.py
```
