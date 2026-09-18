# StitchPartnerAI — Visual Product Search

Photograph a garment-machinery part, get the matching StitchPartner products.
FastAPI + DINOv2 embeddings, with WooCommerce as the source of truth.

```
WooCommerce ──► sync ──► image download ──► DINOv2 embedding ──► vector index
                                                                      │
                          customer photo ──► embedding ──► search ────┘──► results
```

---

## What changed from v1, and why

The previous version had no WooCommerce integration at all. Its index was built
offline on a developer machine and the `.npy` files were committed to the
repository, which is exactly why products added to WooCommerce never appeared in
search: nothing in the running application had ever talked to WooCommerce.

| Area | Before | Now |
|---|---|---|
| Product source | `.npy` committed to git | WooCommerce REST API |
| New products | Manual offline rebuild | Webhook + hourly sync |
| Incremental | Rebuild everything | Only changed images are embedded |
| Product identity | Product **name** string | WooCommerce **product ID** |
| Index storage | In the repo | Persistent volume, atomic writes |
| Startup | Model download blocks boot | Load, warm up, serve |
| Customer uploads | Written to `temp_search/` | Held in memory, never touch disk |
| Admin endpoints | None | Bearer-token protected |
| Errors | `str(e)` returned to customer | Logged server-side, plain message out |

A full breakdown is in [`AUDIT.md`](AUDIT.md).

---

## Setup

Requires Python 3.11 or 3.12.

```bash
git clone <repo> && cd stitchpartner-visual-search
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then fill it in
```

Generate the two secrets:

```bash
openssl rand -hex 32      # ADMIN_API_KEY
openssl rand -hex 32      # WEBHOOK_SECRET
```

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `WOOCOMMERCE_URL` | — | Store base URL, no trailing slash |
| `WOOCOMMERCE_CONSUMER_KEY` | — | REST API key (read-only is enough) |
| `WOOCOMMERCE_CONSUMER_SECRET` | — | REST API secret |
| `ADMIN_API_KEY` | — | Bearer token for `/sync` and `/sync/status` |
| `WEBHOOK_SECRET` | — | Must match the WooCommerce webhook secret |
| `INDEX_PATH` | `./data/index` | Where the index lives — point at a volume |
| `HF_HOME` | `<INDEX_PATH>/../hf-cache` | Model cache — point at the same volume |
| `MODEL_NAME` | `facebook/dinov2-base` | Embedding model |
| `EMBEDDING_VERSION` | `v2` | Bump to invalidate every stored embedding |
| `POOLING` | `cls` | `cls` or `cls_patch` — measure before switching |
| `PREPROCESS_MAX_DIM` | `768` | Longest edge before the model's own resize |
| `TORCH_THREADS` | `0` (all cores) | Intra-op threads |
| `MIN_MATCH_SCORE` | `0.45` | Below this we report no match |
| `STRONG_MATCH_SCORE` | `0.65` | Above this we call it a strong match |
| `MAX_UPLOAD_SIZE` | `10485760` | 10 MB |
| `SYNC_INTERVAL_MINUTES` | `60` | `0` disables the periodic sync |
| `SYNC_ON_STARTUP` | `false` | Leave off — a deploy must not wait on a sync |
| `SEARCH_RATE_LIMIT` | `20` per 60s | Per client IP |
| `CORS_ORIGINS` | `*` | Set to your storefront in production |

Secrets belong in Railway's variable editor or a local `.env`, never in git.

---

## Local development

```bash
uvicorn app:app --reload --port 8000
```

Then open <http://localhost:8000>.

Bring the existing 51-product index across so search works before the first sync:

```bash
python scripts/migrate_legacy_index.py --source product_database
```

Build the index from WooCommerce for the first time:

```bash
python rebuild_index.py
```

---

## WooCommerce configuration

**1. REST API key** — WooCommerce → Settings → Advanced → REST API → Add key.
Permissions: **Read**. Copy the consumer key and secret into your environment.

**2. Webhooks** — WooCommerce → Settings → Advanced → Webhooks. Create three,
all with Delivery URL `https://<your-app>.up.railway.app/webhook/woocommerce`
and Secret set to your `WEBHOOK_SECRET`:

| Topic | Effect |
|---|---|
| `Product created` | Indexes the new product within seconds |
| `Product updated` | Re-embeds if the images changed, removes it if unpublished |
| `Product deleted` | Removes it from the index |

The hourly sync is the safety net: if a webhook is missed or fails, the next sync
reconciles everything anyway.

---

## Initial index creation

```bash
python rebuild_index.py
```

Builds into `<INDEX_PATH>.rebuilding`, validates it, then swaps it into place.
The previous index is kept as `<INDEX_PATH>.previous`. If the build fails at any
point, the live index is left untouched.

```bash
python rebuild_index.py --dry-run              # build and validate only
python rebuild_index.py --min-products 50      # refuse to promote a suspiciously small index
```

---

## Incremental sync

Automatic, via the webhook and the hourly scheduler. To trigger it by hand:

```bash
# Whole catalogue, waiting for the report — the useful mode for debugging
curl -X POST "https://<app>/sync?wait=true" \
     -H "Authorization: Bearer $ADMIN_API_KEY"

# One product
curl -X POST "https://<app>/sync?product_id=103&wait=true" \
     -H "Authorization: Bearer $ADMIN_API_KEY"

# Check on a background sync
curl "https://<app>/sync/status" -H "Authorization: Bearer $ADMIN_API_KEY"
```

The report tells you exactly what happened:

```json
{
  "products_found": 102, "already_indexed": 51, "new_products": 51,
  "updated_products": 0, "removed_products": 0,
  "embeddings_generated": 187, "images_skipped": 0, "errors": []
}
```

A product whose images are unchanged costs one HTTP request and no inference, so
adding one product to a catalogue of a thousand generates one embedding.

---

## Railway deployment

1. Create the service from this repository.
2. **Add a volume** mounted at `/data`. Without it the index and the model cache
   are lost on every deploy, and the first request after each deploy pays for a
   ~350 MB model download.
3. Set the environment variables, including `INDEX_PATH=/data/index` and
   `HF_HOME=/data/hf-cache`.
4. Start command (already in `railway.json`):

   ```bash
   uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
   ```

5. Health check path: `/health`, timeout 300s (the first boot downloads the model).

Keep `--workers 1`. Each worker loads its own copy of the model, and at 80–100
daily users one is plenty.

Never put `rebuild_index.py` in the start command. Run it from
`railway run python rebuild_index.py` or a one-off job.

---

## API

| Route | Auth | Purpose |
|---|---|---|
| `GET /` | — | The search page |
| `GET /health` | — | Model + index status. No inference, no WooCommerce call |
| `POST /search` | Rate limited | `multipart/form-data` with `file`; returns matching products |
| `GET /similar/{product_id}` | — | "You may also like", from stored embeddings |
| `POST /sync` | Bearer | `?product_id=`, `?wait=true` |
| `GET /sync/status` | Bearer | Last report and current state |
| `POST /webhook/woocommerce` | HMAC signature | Product created / updated / deleted |

---

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

21 tests covering index persistence, idempotency, the add/update/remove
lifecycle, corrupt-image handling, upload limits, auth, webhook signatures and
rate limiting. They use a fake WooCommerce and a stub embedder, so they need no
network, no credentials and no model weights.

**Latency**, on the machine that actually matters:

```bash
railway run python scripts/benchmark.py --image sample.jpg --runs 5
```

**Accuracy**, before and after any change to the model, pooling or preprocessing:

```bash
python scripts/evaluate_accuracy.py --dataset tests/dataset
```

Build the dataset as one folder per WooCommerce product ID, plus a `_no_match`
folder of unrelated objects. Cover: exact catalogue image, different angle,
different lighting, phone photo, screenshot, cropped, blurry, several products in
frame, a visually similar different part, and something unrelated.

---

## Troubleshooting

**A new product isn't appearing.**
Check `GET /sync/status` first. Then WooCommerce → Webhooks → your webhook →
Delivery log. A 401 there means `WEBHOOK_SECRET` doesn't match. If the webhook is
fine, run `POST /sync?product_id=<id>&wait=true` and read the `errors` array.

**Every search is slow.**
Run `scripts/benchmark.py` on Railway. If model load appears in the logs on every
request, `HF_HOME` isn't on the volume. If inference is above ~2s, the container
is CPU-starved — check `TORCH_THREADS` and the plan's CPU allocation.

**`/health` says `degraded`.**
`model.loaded: false` means the weights failed to download; check the logs and
the volume's free space. `index.available: false` means the index is empty — run
a sync or `rebuild_index.py`.

**Search returns nothing for images that used to work.**
If you changed `MODEL_NAME`, `POOLING` or `EMBEDDING_VERSION`, every stored
embedding was invalidated on the next boot by design. Run `rebuild_index.py`.

**The index looks inconsistent after a crash.**
`vector_store` refuses to load a mismatched index and logs the counts. Run
`rebuild_index.py` to repair it.

---

## Privacy

Customer photos are read into memory, embedded, and discarded when the request
ends. They are never written to disk, never logged, and never served from a URL.
Only the byte count reaches the logs.
