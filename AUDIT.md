# Audit of the original StitchPartnerAI system

Traced from the code as committed, not assumed. Measurements were taken on a
single-core container, which approximates a small Railway instance.

---

## What was actually there

Four files of substance:

```
app.py                  1,948 lines — 1,620 of them an inline HTML page
visual_search.py          712 lines — model, index and search, all at import time
requirements.txt           50 pinned packages
product_database/          committed .npy and .json index files
```

`product_database/` held:

| File | Contents |
|---|---|
| `woocommerce_embeddings.npy` | 189 × 768 float32 — the live index |
| `woocommerce_image_records.json` | 189 image records across 51 product IDs |
| `woocommerce_product_groups.json` | 51 products with local file paths |
| `embeddings.npy`, `image_names.npy`, `product_groups.json` | 31 vectors, dead — nothing reads them |

The architecture map, as it really was:

```
[developer laptop] → woocommerce_products/*.jpg → embeddings → .npy → git
                                                                       │
                                                                       ▼
                                          app.py loads .npy → /search → results
```

There is no arrow from WooCommerce to the running application.

---

## Finding 1 — the running application never contacted WooCommerce

`grep` for `requests`, `wp-json`, `wc/v3`, `consumer` across `app.py` and
`visual_search.py` returns nothing. No API client, no credentials, no HTTP call.
The `.npy` files were produced by a build script that is not in the repository —
its `local_path` values (`woocommerce_products/141_.../1.jpg`) and the Windows
backslashes in the dead `image_names.npy` show it ran on a developer machine.

**This is the whole answer to "why don't new products appear".** The 51 products
were whatever existed on that laptop when the script last ran. Nothing has
checked since. There was no sync to fix, because there was no sync.

*Fixed by:* `woocommerce.py` (REST client), `sync.py` (incremental reconcile),
`/webhook/woocommerce`, and an hourly scheduler.

---

## Finding 2 — the index rebuilt itself in memory on every boot, and search was fine

`visual_search.py` ran everything at module import: model download, model load,
`np.load`, and a full renormalisation of all 189 vectors. Embeddings were not
*regenerated* on startup, so that particular worry was unfounded — but the model
download was blocking uvicorn from binding a port.

---

## Finding 3 — the 50-second request is not inference

Measured, single core:

| Stage | Measured | Share of a warm request |
|---|---|---|
| PIL decode of a 6.6 MB 4032×3024 JPEG | 139 ms | 12% |
| Resize 4032 px → 224 px | 57 ms | 5% |
| DINOv2-base forward pass, 224 px (257 tokens) | ~650 ms¹ | 83% |
| Vector search, 189 × 768 | **0.014 ms** | 0.001% |
| **Total, warm** | **≈ 850 ms** | |

¹ Measured as the equivalent BLAS GEMM workload (12 layers, d=768, 257 tokens);
ViT inference is over 90% matrix multiplication. Real torch adds framework
overhead but is in the same range.

A warm request should be around one second. Fifty seconds is roughly 60× that,
so the cost is not in the pipeline — it is in the cold start:

- `facebook/dinov2-base` is ~350 MB. With no `HF_HOME` on a persistent volume,
  Railway re-downloads it on every deploy and every container restart.
- Model loading happens at *import*, so uvicorn cannot bind its port until the
  download finishes. There was no `/health` endpoint, so Railway fell back to a
  TCP check and the whole boot appeared as one long stall.
- No warmup, so the first real inference also paid for lazy kernel init.
- 350 MB of fp32 weights on a small container invites swapping, which turns a
  650 ms matmul into tens of seconds.

*Fixed by:* `HF_HOME` on the volume, model load moved into `lifespan` with a
warmup pass, a cheap `/health` for Railway to check, and per-stage timings in the
logs so the real number is visible rather than inferred.

---

## Finding 4 — FAISS was in requirements and never imported

Search was `np.dot` followed by a full `argsort` of every score. At 189 vectors
that is 14 microseconds. Measured scaling:

| Vectors | argsort | argpartition |
|---|---|---|
| 189 | 0.014 ms | 0.013 ms |
| 1,000 | 0.123 ms | 0.117 ms |
| 10,000 | 1.11 ms | 1.03 ms |
| 50,000 | 8.97 ms | 7.76 ms |

Brute force is correct here and stays correct well past 10,000 products. FAISS
would add a dependency, an index-training step and a persistence format to save
a millisecond. It has been dropped from `requirements.txt`; `VectorStore.search`
is a narrow enough seam to swap later if the catalogue reaches six figures.

*Changed:* `argsort` → `argpartition` (O(n) rather than O(n log n)) and `TOP_K`
raised from 10 to 40, because 10 candidate images across 51 products left the
multi-view scoring very little to work with.

---

## Finding 5 — products were keyed by name

`visual_search.search()` discarded `product_id` and returned only
`{"product": name, "score": …}`. `app.py` then looked the product back up with
`woocommerce_products[product_name]`.

Two consequences: renaming a product in WooCommerce silently broke its URL and
image, and two products sharing a name collided into one entry — the first
record encountered won, and the second became unreachable.

*Fixed by:* product ID is the key throughout. Names are display data.

---

## Finding 6 — uploads

`app.py` validated `Path(file.filename).suffix` and nothing else, then streamed
the body straight to `temp_search/` with `shutil.copyfileobj`.

- The filename is client-supplied. `payload.jpg` containing anything at all
  passed the check.
- No size limit anywhere. A 500 MB upload was written to the container disk.
- No pixel limit, so a decompression-bomb PNG could exhaust memory.
- Cleanup was in a `finally`, which was correct — but if the process was killed
  mid-request the file survived, and nothing ever swept the directory.
- `Image.open(...)` ignored EXIF orientation, so a portrait phone photo reached
  the model on its side.

*Fixed by:* the upload never touches disk. It is read in bounded chunks into
memory, rejected past `MAX_UPLOAD_SIZE`, and the format is determined by Pillow
from the bytes themselves. `Image.MAX_IMAGE_PIXELS` guards decompression bombs,
`ImageOps.exif_transpose` rotates the photo the right way up, and JPEG draft-mode
decoding cuts the decode-and-resize cost from 196 ms to 80 ms.

---

## Finding 7 — errors leaked internals

```python
return {"success": False, "message": "Visual search engine error: " + str(e)}
```

Exception text went straight to the customer. `print()` was the only logging, so
nothing was levelled, timestamped or filterable.

*Fixed by:* structured logging with per-stage timings, and one plain sentence to
the customer.

---

## Finding 8 — no protected endpoints, because there were no endpoints

`/` and `/search`. No `/health`, no `/sync`, no webhook, no auth, no rate limit,
no CORS policy. Nothing was exposed that shouldn't have been, but nothing
operable existed either.

---

## Finding 9 — the frontend

1,620 lines of HTML inlined in `app.py`. It worked, and the responsive
breakpoints were real. What it lacked: no client-side compression (a 8 MB photo
went up whole — roughly 40 seconds on slow 4G before the AI even started), no
request timeout, no retry, no `capture` attribute for the camera, no skeleton
loaders, and one indefinite spinner regardless of what stage the request was in.

*Fixed by:* a separate `static/index.html` (22 KB, no external assets or
webfonts) that downscales to 1024 px and re-encodes to JPEG before upload —
typically 8 MB to about 200 KB — with a 45-second timeout, a retry path, staged
progress messages, a "still working" note after 8 seconds, skeleton cards, lazy
result images and a bottom-sheet camera/gallery picker.

---

## What was kept

Several things worked and had no evidence against them:

- **DINOv2-base** — kept. It is a strong choice for part-level visual similarity
  and it had already been accuracy-tested against the first batch.
- **CLS-token pooling** — kept as the default, so the existing 189 embeddings
  remain valid. `POOLING=cls_patch` is available but must be A/B measured with
  `scripts/evaluate_accuracy.py` before anyone switches it on.
- **Multi-view scoring with the consistency bonus** — kept, weights unchanged.
  A product matching on several of its photos genuinely is a better answer.
- **The similarity floor** — kept at 0.45, with the copy reworded for customers.
- **The existing 189 embeddings** — carried across by
  `scripts/migrate_legacy_index.py`, so search keeps working from the first boot
  rather than going dark until the first sync completes.

---

## Still to be done on real infrastructure

These need the live store and the Railway container, and cannot be closed here:

1. Run `scripts/benchmark.py` on Railway and confirm the warm search lands under
   3 seconds. The number above is a single-core estimate, not a Railway
   measurement.
2. Build the labelled accuracy dataset and record baseline top-1/3/5 figures
   before tuning `MIN_MATCH_SCORE`. The current 0.45 is inherited, not measured.
3. Run the product #103 add / change-image / unpublish cycle against the real
   store. The logic is covered by tests against a fake WooCommerce, which is not
   the same as the real API.
4. Device testing across the listed viewports and browsers.
5. Decide on multiple-object handling. Right now clutter is handled only by the
   confidence floor and the "one part at a time" prompt; object detection was not
   added, because at 100 products it would be complexity without measured benefit.
