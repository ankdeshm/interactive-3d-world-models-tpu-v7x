# Appendix: Complete 266-Case Benchmark Tables, Sharding Trade-offs, and Caveats

This document records all 266 verified empirical benchmark rows from [`results/`](./results/), along with TorchTPU execution details, configurations that regressed or missed frame-rate targets, and measurement caveats for [`README.md`](./README.md).

---

## 1. Execution Methodology & Important Measurement Caveats

1. **HBM weight staging and layer-scaled DiT + VAE timing:**
   - [`scripts/run_interactive_3d_worlds_v7x.py`](./scripts/run_interactive_3d_worlds_v7x.py) stages the checkpoint files (`28.13 GiB` `Matrix-Game-4.0`, `19.97 GiB` `Matrix-Game-2.0`, `76.96 GiB` `HunyuanWorld-Voyager` via zero-copy `mmap`, and `1.45 GiB` `HunyuanWorld-1` rank-128 adapters) onto TPU HBM3e (`10.2 GiB`, `6.8 GiB`, `26.0 GiB`, and `1.45 GiB` active per replica before `Ulysses × Ring` sharding and `FP8` halving).
   - Each `/step` call runs `6` architecturally matched causal-KV or full-window DiT blocks (`d_model`, `n_heads`, `ffn_dim`, and optional action cross-attention) scaled to full model depth (`30`, `57`, or `60` layers × `denoise_steps`), plus a 2-stage `conv2d + pixel_shuffle` VAE decode pass and hardware-calibrated 8-TensorCore `torch.distributed` ICI `all_reduce` latency.
2. **Suite 3 (`concurrent_player_sessions_sla.json`) RPC-timing caveat:**
   - In Suite 1 (`152` rows) and Suite 2 (`66` rows), chunk latency and stream FPS are computed from `chunk_interactive_latency_ms` (`per_step_s * num_steps + vae_decode_s`, properly scaled across all `30–60` layers and `4–20` denoising steps).
   - In Suite 3 (`48` rows, `lines 1332–1354` of `run_interactive_3d_worlds_v7x.py`), `_player_session` recorded client wall-clock HTTP RTT around a single `/step` RPC (`6` sampled layers × `1` pass + `1` VAE pass, without multiplying by `(n_layers / 6) * num_steps`) while concurrently streaming 8-frame chunks to `video-encode-svc:8090/encode`. Consequently, Suite 3 `chunk_to_chunk_p50_ms` (`5.67–26.79 ms`) measures **multi-host HTTP dispatch + single-sample TPU kernel + async CPU encoder concurrency**, NOT full-depth multi-step DiT generation. For true full-depth multi-step `16 FPS` / `24 FPS` capability, refer to Suite 1 and Suite 2 (where `HunyuanWorld-Voyager` at `720p` 8-step reaches `19.90–20.35 FPS` and misses `24 FPS`, rather than passing at 32 streams).
3. **Suite 2 `aggregate_slice_fps` includes synchronous MP4 encoding:**
   - In Suite 2, every measured request sets `encode_mode="remote"` (`119–126 ms` round-trip to the 2-pod `video-encode-svc`), so `aggregate_slice_fps` reflects end-to-end compressed MP4 delivery rather than raw DiT latent throughput.

---

## 2. What Worked vs. What Regressed or Missed Frame-Rate Targets

- **Causal rolling KV-cache reuse (`kv_cache_reuse=True`):** Restricting Q projections and attention outputs to the newly generated chunk (`4`, `8`, or `16` frames) while attending over cached historical K/V states reduced compiled `fp8` chunk latency by **`2.67×–3.51×`** on default 8-frame `480p` streams (`184.75 -> 52.69 ms` on `Matrix-Game-4.0`, `86.78 -> 30.02 ms` on `Matrix-Game-2.0`, `788.73 -> 295.71 ms` on `HunyuanWorld-Voyager`) and up to **`4.81×–5.88×`** on 4-frame `720p` chunks (`910.68 -> 189.23 ms` on `Matrix-Game-4.0`, `435.09 -> 73.97 ms` at `4` steps).
- **Uncompiled eager `fp8_e4m3fn` regressed on smaller/adapter models:** Without `torch.compile`, eager `fp8` was slower than eager `bf16` on `Matrix-Game-2.0` (`211.35 ms` vs. `189.22 ms` full-window; `179.21 ms` vs. `145.72 ms` KV-cache) and `HunyuanWorld-1` (`579.62 ms` vs. `488.13 ms` full-window) and slower on KV-cached `Matrix-Game-4.0` (`184.07 ms` vs. `173.00 ms`) and `HunyuanWorld-Voyager` (`706.26 ms` vs. `598.26 ms`) because short causal-chunk GEMMs (`m_rows`) either fell back to `bf16` casts or paid unfused `torch._scaled_mm` dispatch overhead. With `torch.compile`, `fp8` consistently outperformed compiled `bf16` across all 4 models (`1.10×–1.18×`).
- **`HunyuanWorld-Voyager` (`13B`, `60L`, `8` steps) frame-rate misses in Suite 2:**
  - At `480p` (`8` steps), `U=2` (`20.23–21.15 FPS`), `U=4` (`22.16–22.31 FPS`), and 1-host `U=8` (`23.79 FPS`) met `16 FPS` but missed `24 FPS`; adding ring KV sharding (`U=4 × R=2`, `264.43–289.56 ms` = `27.62–30.25 FPS`) or multi-host `U=8` (`24.73–25.41 FPS`) crossed `24 FPS`.
  - At `720p` (`8` steps), `U=2` (`7.21–7.35 FPS`), `U=4` (`12.05–12.29 FPS`), and `U=8` (`15.45–15.84 FPS`) missed both `16 FPS` and `24 FPS`. Hybrid `U=4 × R=2` (`393.04–401.98 ms` = `19.90–20.35 FPS`) met `16 FPS` by sharding the historical KV cache (`s_hist`) across the ring dimension, while reaching `>24 FPS` at `720p` required 4-step distillation (`303.35 ms` = `26.37 FPS` in Suite 1).
- **20-step denoising on interactive streams:** Increasing denoising steps from `4` to `20` dropped `720p` 8-frame `Matrix-Game-4.0` to `502.60 ms` (`15.92 FPS`), `HunyuanWorld-Voyager` to `1,450.68 ms` (`5.51 FPS`), and `HunyuanWorld-1` to `1,417.26 ms` (`5.64 FPS`), confirming that 4–8 step distilled schedules are mandatory for real-time interaction.

---

## 3. Suite 1A: Precision, Compilation & KV-Cache Ablation (32 Cases, `1` Host `U=4`, `480p`, `8`-Frame Chunk)

Raw JSON: [`results/causal_kv_vs_fullwindow_ablation.json`](./results/causal_kv_vs_fullwindow_ablation.json).

| Model | Precision | Execution | KV Reuse | Steps | Step (ms) | VAE (ms) | Chunk Latency (ms) | Stream FPS | Local / Disagg MP4 (ms) | Speedup vs. Eager `BF16` No-KV |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| `Matrix-Game-4.0` | `BF16` | `EAGER` | `NO` | 4 | 66.43 | 54.29 | 317.86 | 25.17 | — | 1.00× |
| `Matrix-Game-4.0` | `BF16` | `EAGER` | `YES` | 4 | 41.84 | 4.20 | 173.00 | 46.24 | — | 1.84× |
| `Matrix-Game-4.0` | `BF16` | `COMPILED` | `NO` | 4 | 50.43 | 0.38 | 202.13 | 39.58 | — | 1.57× |
| `Matrix-Game-4.0` | `BF16` | `COMPILED` | `YES` | 4 | 15.38 | 0.37 | 61.90 | 129.23 | — | 5.14× |
| `Matrix-Game-4.0` | `FP8` | `EAGER` | `NO` | 4 | 58.42 | 4.22 | 237.43 | 33.69 | — | 1.34× |
| `Matrix-Game-4.0` | `FP8` | `EAGER` | `YES` | 4 | 45.02 | 3.90 | 184.07 | 43.46 | — | 1.73× |
| `Matrix-Game-4.0` | `FP8` | `COMPILED` | `NO` | 4 | 46.09 | 0.38 | 184.75 | 43.30 | — | 1.72× |
| `Matrix-Game-4.0` | `FP8` | `COMPILED` | `YES` | 4 | 13.09 | 0.38 | 52.69 | 151.83 | 141.4 / 126.0 | 6.03× |
| `Matrix-Game-2.0` | `BF16` | `EAGER` | `NO` | 4 | 46.38 | 3.65 | 189.22 | 42.28 | — | 1.00× |
| `Matrix-Game-2.0` | `BF16` | `EAGER` | `YES` | 4 | 35.45 | 3.71 | 145.72 | 54.90 | — | 1.30× |
| `Matrix-Game-2.0` | `BF16` | `COMPILED` | `NO` | 4 | 22.12 | 0.34 | 88.86 | 90.03 | — | 2.13× |
| `Matrix-Game-2.0` | `BF16` | `COMPILED` | `YES` | 4 | 8.17 | 0.45 | 33.12 | 241.55 | — | 5.71× |
| `Matrix-Game-2.0` | `FP8` | `EAGER` | `NO` | 4 | 51.80 | 3.97 | 211.35 | 37.85 | — | 0.90× |
| `Matrix-Game-2.0` | `FP8` | `EAGER` | `YES` | 4 | 43.69 | 4.00 | 179.21 | 44.64 | — | 1.06× |
| `Matrix-Game-2.0` | `FP8` | `COMPILED` | `NO` | 4 | 21.61 | 0.44 | 86.78 | 92.19 | — | 2.18× |
| `Matrix-Game-2.0` | `FP8` | `COMPILED` | `YES` | 4 | 7.41 | 0.38 | 30.02 | 266.52 | 137.8 / 119.4 | 6.30× |
| `HunyuanWorld-Voyager` | `BF16` | `EAGER` | `NO` | 8 | 137.97 | 3.44 | 1107.32 | 7.22 | — | 1.00× |
| `HunyuanWorld-Voyager` | `BF16` | `EAGER` | `YES` | 8 | 74.30 | 3.96 | 598.26 | 13.37 | — | 1.85× |
| `HunyuanWorld-Voyager` | `BF16` | `COMPILED` | `NO` | 8 | 114.52 | 0.38 | 916.53 | 8.73 | — | 1.21× |
| `HunyuanWorld-Voyager` | `BF16` | `COMPILED` | `YES` | 8 | 41.75 | 0.39 | 334.37 | 23.93 | — | 3.31× |
| `HunyuanWorld-Voyager` | `FP8` | `EAGER` | `NO` | 8 | 128.53 | 3.96 | 1032.11 | 7.75 | — | 1.07× |
| `HunyuanWorld-Voyager` | `FP8` | `EAGER` | `YES` | 8 | 87.83 | 3.60 | 706.26 | 11.33 | — | 1.57× |
| `HunyuanWorld-Voyager` | `FP8` | `COMPILED` | `NO` | 8 | 98.54 | 0.45 | 788.73 | 10.14 | — | 1.40× |
| `HunyuanWorld-Voyager` | `FP8` | `COMPILED` | `YES` | 8 | 36.92 | 0.38 | 295.71 | 27.05 | 135.4 / 123.5 | 3.74× |
| `HunyuanWorld-1` | `BF16` | `EAGER` | `NO` | 8 | 60.48 | 3.92 | 488.13 | 16.39 | — | 1.00× |
| `HunyuanWorld-1` | `BF16` | `EAGER` | `YES` | 8 | 52.92 | 3.63 | 427.02 | 18.73 | — | 1.14× |
| `HunyuanWorld-1` | `BF16` | `COMPILED` | `NO` | 8 | 44.90 | 0.36 | 359.57 | 22.25 | — | 1.36× |
| `HunyuanWorld-1` | `BF16` | `COMPILED` | `YES` | 8 | 30.15 | 0.35 | 241.56 | 33.12 | — | 2.02× |
| `HunyuanWorld-1` | `FP8` | `EAGER` | `NO` | 8 | 71.98 | 3.75 | 579.62 | 13.80 | — | 0.84× |
| `HunyuanWorld-1` | `FP8` | `EAGER` | `YES` | 8 | 66.05 | 3.62 | 532.23 | 15.03 | — | 0.92× |
| `HunyuanWorld-1` | `FP8` | `COMPILED` | `NO` | 8 | 37.15 | 0.36 | 297.57 | 26.88 | — | 1.64× |
| `HunyuanWorld-1` | `FP8` | `COMPILED` | `YES` | 8 | 27.04 | 0.41 | 216.73 | 36.91 | 134.0 / 122.8 | 2.25× |

## 4. Suite 1B: Resolution, Chunk Size (`4, 8, 16f`), and Denoise Steps (`4, 8, 20`) Sweep (120 Cases, Compiled `FP8`, `1` Host `U=4`)

| Model | Res | Chunk (`Window`) | Steps | KV Reuse | Chunk Latency (ms) | Stream FPS | KV Speedup (Same Config) | `≥16 FPS` | `≥24 FPS` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | :--- | :--- |
| `Matrix-Game-4.0` | `480p` | `4f` (`64f`) | 4 | `NO` | 186.34 | 21.47 | — | PASS | MISS |
| `Matrix-Game-4.0` | `480p` | `4f` (`64f`) | 4 | `YES` | 42.21 | 94.77 | 4.41× | PASS | PASS |
| `Matrix-Game-4.0` | `480p` | `4f` (`64f`) | 8 | `NO` | 406.62 | 9.84 | — | MISS | MISS |
| `Matrix-Game-4.0` | `480p` | `4f` (`64f`) | 8 | `YES` | 129.40 | 30.91 | 3.14× | PASS | PASS |
| `Matrix-Game-4.0` | `480p` | `4f` (`64f`) | 20 | `YES` | 200.97 | 19.90 | — | PASS | MISS |
| `Matrix-Game-4.0` | `480p` | `8f` (`64f`) | 4 | `NO` | 183.23 | 43.66 | — | PASS | PASS |
| `Matrix-Game-4.0` | `480p` | `8f` (`64f`) | 4 | `YES` | 50.32 | 158.98 | 3.64× | PASS | PASS |
| `Matrix-Game-4.0` | `480p` | `8f` (`64f`) | 8 | `NO` | 411.51 | 19.44 | — | PASS | MISS |
| `Matrix-Game-4.0` | `480p` | `8f` (`64f`) | 8 | `YES` | 133.84 | 59.77 | 3.07× | PASS | PASS |
| `Matrix-Game-4.0` | `480p` | `8f` (`64f`) | 20 | `YES` | 310.47 | 25.77 | — | PASS | PASS |
| `Matrix-Game-4.0` | `480p` | `16f` (`64f`) | 4 | `NO` | 188.34 | 84.95 | — | PASS | PASS |
| `Matrix-Game-4.0` | `480p` | `16f` (`64f`) | 4 | `YES` | 98.50 | 162.44 | 1.91× | PASS | PASS |
| `Matrix-Game-4.0` | `480p` | `16f` (`64f`) | 8 | `NO` | 406.90 | 39.32 | — | PASS | PASS |
| `Matrix-Game-4.0` | `480p` | `16f` (`64f`) | 8 | `YES` | 190.18 | 84.13 | 2.14× | PASS | PASS |
| `Matrix-Game-4.0` | `480p` | `16f` (`64f`) | 20 | `YES` | 399.31 | 40.07 | — | PASS | PASS |
| `Matrix-Game-4.0` | `720p` | `4f` (`64f`) | 4 | `NO` | 435.09 | 9.19 | — | MISS | MISS |
| `Matrix-Game-4.0` | `720p` | `4f` (`64f`) | 4 | `YES` | 73.97 | 54.08 | 5.88× | PASS | PASS |
| `Matrix-Game-4.0` | `720p` | `4f` (`64f`) | 8 | `NO` | 910.68 | 4.39 | — | MISS | MISS |
| `Matrix-Game-4.0` | `720p` | `4f` (`64f`) | 8 | `YES` | 189.23 | 21.14 | 4.81× | PASS | MISS |
| `Matrix-Game-4.0` | `720p` | `4f` (`64f`) | 20 | `YES` | 362.51 | 11.03 | — | MISS | MISS |
| `Matrix-Game-4.0` | `720p` | `8f` (`64f`) | 4 | `NO` | 431.33 | 18.55 | — | PASS | MISS |
| `Matrix-Game-4.0` | `720p` | `8f` (`64f`) | 4 | `YES` | 104.00 | 76.92 | 4.15× | PASS | PASS |
| `Matrix-Game-4.0` | `720p` | `8f` (`64f`) | 8 | `NO` | 897.39 | 8.91 | — | MISS | MISS |
| `Matrix-Game-4.0` | `720p` | `8f` (`64f`) | 8 | `YES` | 248.79 | 32.16 | 3.61× | PASS | PASS |
| `Matrix-Game-4.0` | `720p` | `8f` (`64f`) | 20 | `YES` | 502.60 | 15.92 | — | MISS | MISS |
| `Matrix-Game-4.0` | `720p` | `16f` (`64f`) | 4 | `NO` | 455.92 | 35.09 | — | PASS | PASS |
| `Matrix-Game-4.0` | `720p` | `16f` (`64f`) | 4 | `YES` | 169.26 | 94.53 | 2.69× | PASS | PASS |
| `Matrix-Game-4.0` | `720p` | `16f` (`64f`) | 8 | `NO` | 930.37 | 17.20 | — | PASS | MISS |
| `Matrix-Game-4.0` | `720p` | `16f` (`64f`) | 8 | `YES` | 370.98 | 43.13 | 2.51× | PASS | PASS |
| `Matrix-Game-4.0` | `720p` | `16f` (`64f`) | 20 | `YES` | 846.37 | 18.90 | — | PASS | MISS |
| `Matrix-Game-2.0` | `480p` | `4f` (`64f`) | 4 | `NO` | 86.95 | 46.01 | — | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `4f` (`64f`) | 4 | `YES` | 26.79 | 149.28 | 3.25× | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `4f` (`64f`) | 8 | `NO` | 211.98 | 18.87 | — | PASS | MISS |
| `Matrix-Game-2.0` | `480p` | `4f` (`64f`) | 8 | `YES` | 89.69 | 44.60 | 2.36× | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `4f` (`64f`) | 20 | `YES` | 122.41 | 32.68 | — | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `8f` (`64f`) | 4 | `NO` | 86.56 | 92.42 | — | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `8f` (`64f`) | 4 | `YES` | 28.08 | 284.85 | 3.08× | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `8f` (`64f`) | 8 | `NO` | 203.95 | 39.22 | — | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `8f` (`64f`) | 8 | `YES` | 95.82 | 83.49 | 2.13× | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `8f` (`64f`) | 20 | `YES` | 221.04 | 36.19 | — | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `16f` (`64f`) | 4 | `NO` | 87.69 | 182.46 | — | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `16f` (`64f`) | 4 | `YES` | 41.54 | 385.19 | 2.11× | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `16f` (`64f`) | 8 | `NO` | 212.81 | 75.19 | — | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `16f` (`64f`) | 8 | `YES` | 123.30 | 129.76 | 1.73× | PASS | PASS |
| `Matrix-Game-2.0` | `480p` | `16f` (`64f`) | 20 | `YES` | 194.58 | 82.23 | — | PASS | PASS |
| `Matrix-Game-2.0` | `720p` | `4f` (`64f`) | 4 | `NO` | 217.02 | 18.43 | — | PASS | MISS |
| `Matrix-Game-2.0` | `720p` | `4f` (`64f`) | 4 | `YES` | 41.60 | 96.16 | 5.22× | PASS | PASS |
| `Matrix-Game-2.0` | `720p` | `4f` (`64f`) | 8 | `NO` | 487.33 | 8.21 | — | MISS | MISS |
| `Matrix-Game-2.0` | `720p` | `4f` (`64f`) | 8 | `YES` | 105.26 | 38.00 | 4.63× | PASS | PASS |
| `Matrix-Game-2.0` | `720p` | `4f` (`64f`) | 20 | `YES` | 200.30 | 19.97 | — | PASS | MISS |
| `Matrix-Game-2.0` | `720p` | `8f` (`64f`) | 4 | `NO` | 215.66 | 37.10 | — | PASS | PASS |
| `Matrix-Game-2.0` | `720p` | `8f` (`64f`) | 4 | `YES` | 54.57 | 146.61 | 3.95× | PASS | PASS |
| `Matrix-Game-2.0` | `720p` | `8f` (`64f`) | 8 | `NO` | 472.45 | 16.93 | — | PASS | MISS |
| `Matrix-Game-2.0` | `720p` | `8f` (`64f`) | 8 | `YES` | 146.49 | 54.61 | 3.23× | PASS | PASS |
| `Matrix-Game-2.0` | `720p` | `8f` (`64f`) | 20 | `YES` | 259.42 | 30.84 | — | PASS | PASS |
| `Matrix-Game-2.0` | `720p` | `16f` (`64f`) | 4 | `NO` | 216.71 | 73.83 | — | PASS | PASS |
| `Matrix-Game-2.0` | `720p` | `16f` (`64f`) | 4 | `YES` | 84.42 | 189.53 | 2.57× | PASS | PASS |
| `Matrix-Game-2.0` | `720p` | `16f` (`64f`) | 8 | `NO` | 479.26 | 33.39 | — | PASS | PASS |
| `Matrix-Game-2.0` | `720p` | `16f` (`64f`) | 8 | `YES` | 194.50 | 82.26 | 2.46× | PASS | PASS |
| `Matrix-Game-2.0` | `720p` | `16f` (`64f`) | 20 | `YES` | 409.53 | 39.07 | — | PASS | PASS |
| `HunyuanWorld-Voyager` | `480p` | `4f` (`49f`) | 4 | `NO` | 394.39 | 10.14 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `480p` | `4f` (`49f`) | 4 | `YES` | 106.04 | 37.72 | 3.72× | PASS | PASS |
| `HunyuanWorld-Voyager` | `480p` | `4f` (`49f`) | 8 | `NO` | 868.60 | 4.61 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `480p` | `4f` (`49f`) | 8 | `YES` | 288.60 | 13.86 | 3.01× | MISS | MISS |
| `HunyuanWorld-Voyager` | `480p` | `4f` (`49f`) | 20 | `YES` | 498.36 | 8.03 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `480p` | `8f` (`49f`) | 4 | `NO` | 414.03 | 19.32 | — | PASS | MISS |
| `HunyuanWorld-Voyager` | `480p` | `8f` (`49f`) | 4 | `YES` | 143.65 | 55.69 | 2.88× | PASS | PASS |
| `HunyuanWorld-Voyager` | `480p` | `8f` (`49f`) | 8 | `NO` | 874.45 | 9.15 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `480p` | `8f` (`49f`) | 8 | `YES` | 354.65 | 22.56 | 2.47× | PASS | MISS |
| `HunyuanWorld-Voyager` | `480p` | `8f` (`49f`) | 20 | `YES` | 937.16 | 8.54 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `480p` | `16f` (`64f`) | 4 | `NO` | 476.73 | 33.56 | — | PASS | PASS |
| `HunyuanWorld-Voyager` | `480p` | `16f` (`64f`) | 4 | `YES` | 201.77 | 79.30 | 2.36× | PASS | PASS |
| `HunyuanWorld-Voyager` | `480p` | `16f` (`64f`) | 8 | `NO` | 1011.91 | 15.81 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `480p` | `16f` (`64f`) | 8 | `YES` | 483.48 | 33.09 | 2.09× | PASS | PASS |
| `HunyuanWorld-Voyager` | `480p` | `16f` (`64f`) | 20 | `YES` | 963.77 | 16.60 | — | PASS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `4f` (`49f`) | 4 | `NO` | 1001.98 | 3.99 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `4f` (`49f`) | 4 | `YES` | 202.61 | 19.74 | 4.95× | PASS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `4f` (`49f`) | 8 | `NO` | 2082.12 | 1.92 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `4f` (`49f`) | 8 | `YES` | 462.83 | 8.64 | 4.50× | MISS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `4f` (`49f`) | 20 | `YES` | 1000.59 | 4.00 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `8f` (`49f`) | 4 | `NO` | 948.00 | 8.44 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `8f` (`49f`) | 4 | `YES` | 303.35 | 26.37 | 3.13× | PASS | PASS |
| `HunyuanWorld-Voyager` | `720p` | `8f` (`49f`) | 8 | `NO` | 1961.46 | 4.08 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `8f` (`49f`) | 8 | `YES` | 670.24 | 11.94 | 2.93× | MISS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `8f` (`49f`) | 20 | `YES` | 1450.68 | 5.51 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `16f` (`64f`) | 4 | `NO` | 1319.72 | 12.12 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `16f` (`64f`) | 4 | `YES` | 593.60 | 26.95 | 2.22× | PASS | PASS |
| `HunyuanWorld-Voyager` | `720p` | `16f` (`64f`) | 8 | `NO` | 2694.97 | 5.94 | — | MISS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `16f` (`64f`) | 8 | `YES` | 1239.66 | 12.91 | 2.17× | MISS | MISS |
| `HunyuanWorld-Voyager` | `720p` | `16f` (`64f`) | 20 | `YES` | 3015.27 | 5.31 | — | MISS | MISS |
| `HunyuanWorld-1` | `480p` | `4f` (`16f`) | 4 | `NO` | 149.26 | 26.80 | — | PASS | PASS |
| `HunyuanWorld-1` | `480p` | `4f` (`16f`) | 4 | `YES` | 71.32 | 56.08 | 2.09× | PASS | PASS |
| `HunyuanWorld-1` | `480p` | `4f` (`16f`) | 8 | `NO` | 369.99 | 10.81 | — | MISS | MISS |
| `HunyuanWorld-1` | `480p` | `4f` (`16f`) | 8 | `YES` | 213.29 | 18.75 | 1.73× | PASS | MISS |
| `HunyuanWorld-1` | `480p` | `4f` (`16f`) | 20 | `YES` | 341.04 | 11.73 | — | MISS | MISS |
| `HunyuanWorld-1` | `480p` | `8f` (`32f`) | 4 | `NO` | 301.11 | 26.57 | — | PASS | PASS |
| `HunyuanWorld-1` | `480p` | `8f` (`32f`) | 4 | `YES` | 123.93 | 64.55 | 2.43× | PASS | PASS |
| `HunyuanWorld-1` | `480p` | `8f` (`32f`) | 8 | `NO` | 689.47 | 11.60 | — | MISS | MISS |
| `HunyuanWorld-1` | `480p` | `8f` (`32f`) | 8 | `YES` | 346.57 | 23.08 | 1.99× | PASS | MISS |
| `HunyuanWorld-1` | `480p` | `8f` (`32f`) | 20 | `YES` | 630.23 | 12.69 | — | MISS | MISS |
| `HunyuanWorld-1` | `480p` | `16f` (`64f`) | 4 | `NO` | 637.36 | 25.10 | — | PASS | PASS |
| `HunyuanWorld-1` | `480p` | `16f` (`64f`) | 4 | `YES` | 258.92 | 61.80 | 2.46× | PASS | PASS |
| `HunyuanWorld-1` | `480p` | `16f` (`64f`) | 8 | `NO` | 1342.91 | 11.91 | — | MISS | MISS |
| `HunyuanWorld-1` | `480p` | `16f` (`64f`) | 8 | `YES` | 579.19 | 27.62 | 2.32× | PASS | PASS |
| `HunyuanWorld-1` | `480p` | `16f` (`64f`) | 20 | `YES` | 1246.24 | 12.84 | — | MISS | MISS |
| `HunyuanWorld-1` | `720p` | `4f` (`16f`) | 4 | `NO` | 344.71 | 11.60 | — | MISS | MISS |
| `HunyuanWorld-1` | `720p` | `4f` (`16f`) | 4 | `YES` | 133.37 | 29.99 | 2.58× | PASS | PASS |
| `HunyuanWorld-1` | `720p` | `4f` (`16f`) | 8 | `NO` | 782.94 | 5.11 | — | MISS | MISS |
| `HunyuanWorld-1` | `720p` | `4f` (`16f`) | 8 | `YES` | 316.58 | 12.64 | 2.47× | MISS | MISS |
| `HunyuanWorld-1` | `720p` | `4f` (`16f`) | 20 | `YES` | 655.25 | 6.10 | — | MISS | MISS |
| `HunyuanWorld-1` | `720p` | `8f` (`32f`) | 4 | `NO` | 717.98 | 11.14 | — | MISS | MISS |
| `HunyuanWorld-1` | `720p` | `8f` (`32f`) | 4 | `YES` | 295.61 | 27.06 | 2.43× | PASS | PASS |
| `HunyuanWorld-1` | `720p` | `8f` (`32f`) | 8 | `NO` | 1509.99 | 5.30 | — | MISS | MISS |
| `HunyuanWorld-1` | `720p` | `8f` (`32f`) | 8 | `YES` | 647.97 | 12.35 | 2.33× | MISS | MISS |
| `HunyuanWorld-1` | `720p` | `8f` (`32f`) | 20 | `YES` | 1417.26 | 5.64 | — | MISS | MISS |
| `HunyuanWorld-1` | `720p` | `16f` (`64f`) | 4 | `NO` | 1656.67 | 9.66 | — | MISS | MISS |
| `HunyuanWorld-1` | `720p` | `16f` (`64f`) | 4 | `YES` | 997.87 | 16.03 | 1.66× | PASS | MISS |
| `HunyuanWorld-1` | `720p` | `16f` (`64f`) | 8 | `NO` | 3362.07 | 4.76 | — | MISS | MISS |
| `HunyuanWorld-1` | `720p` | `16f` (`64f`) | 8 | `YES` | 2043.07 | 7.83 | 1.65× | MISS | MISS |
| `HunyuanWorld-1` | `720p` | `16f` (`64f`) | 20 | `YES` | 4939.40 | 3.24 | — | MISS | MISS |

---

## 5. Suite 2: Multi-Host Sharding & Topology Scaling (66 Cases, `1` to `4` Hosts, Compiled `FP8` + KV-Cache, `8`-Frame Chunk)

Raw JSON: [`results/sharding_topology_scaling.json`](./results/sharding_topology_scaling.json).

| Model | Topology / Sharding | Hosts (`TCs`) | `DP × U × R` | Res | Steps | Median Chunk (ms) | Per-Stream FPS | Aggregate Slice FPS | `≥16 FPS` | `≥24 FPS` |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| `Matrix-Game-4.0` | 1-Host 8-TC Intra-Chip DP (DP=4, U=2, R=1) | 1 (`8`) | `4×2×1` | `480p` | 4 | 79.81 | 100.23 | 110.4 | PASS | PASS |
| `Matrix-Game-4.0` | 1-Host 8-TC Intra-Chip DP (DP=4, U=2, R=1) | 1 (`8`) | `4×2×1` | `720p` | 4 | 167.29 | 47.83 | 56.1 | PASS | PASS |
| `Matrix-Game-4.0` | 1-Host 8-TC SubHost DP (DP=2, U=4, R=1) | 1 (`8`) | `2×4×1` | `480p` | 4 | 67.80 | 118.02 | 56.2 | PASS | PASS |
| `Matrix-Game-4.0` | 1-Host 8-TC SubHost DP (DP=2, U=4, R=1) | 1 (`8`) | `2×4×1` | `720p` | 4 | 121.03 | 66.10 | 34.9 | PASS | PASS |
| `Matrix-Game-4.0` | 1-Host 8-TC Full-Host Ulysses (DP=1, U=8, R=1) | 1 (`8`) | `1×8×1` | `480p` | 4 | 69.92 | 114.41 | 28.8 | PASS | PASS |
| `Matrix-Game-4.0` | 1-Host 8-TC Full-Host Ulysses (DP=1, U=8, R=1) | 1 (`8`) | `1×8×1` | `720p` | 4 | 104.60 | 76.48 | 17.8 | PASS | PASS |
| `Matrix-Game-4.0` | 1-Host 8-TC Hybrid U4xR2 (DP=1, U=4, R=2) | 1 (`8`) | `1×4×2` | `480p` | 4 | 62.25 | 128.52 | 28.9 | PASS | PASS |
| `Matrix-Game-4.0` | 1-Host 8-TC Hybrid U4xR2 (DP=1, U=4, R=2) | 1 (`8`) | `1×4×2` | `720p` | 4 | 82.15 | 97.39 | 17.4 | PASS | PASS |
| `Matrix-Game-4.0` | 2-Host 16-TC Intra-Chip DP (DP=8, U=2, R=1) | 2 (`16`) | `8×2×1` | `480p` | 4 | 84.84 | 94.30 | 176.4 | PASS | PASS |
| `Matrix-Game-4.0` | 2-Host 16-TC Intra-Chip DP (DP=8, U=2, R=1) | 2 (`16`) | `8×2×1` | `720p` | 4 | 171.06 | 46.77 | 108.7 | PASS | PASS |
| `Matrix-Game-4.0` | 2-Host 16-TC SubHost DP (DP=4, U=4, R=1) | 2 (`16`) | `4×4×1` | `480p` | 4 | 67.81 | 117.98 | 98.9 | PASS | PASS |
| `Matrix-Game-4.0` | 2-Host 16-TC SubHost DP (DP=4, U=4, R=1) | 2 (`16`) | `4×4×1` | `720p` | 4 | 123.94 | 64.56 | 66.0 | PASS | PASS |
| `Matrix-Game-4.0` | 2-Host 16-TC Host-SP DP (DP=2, U=8, R=1) | 2 (`16`) | `2×8×1` | `480p` | 4 | 69.91 | 114.45 | 56.6 | PASS | PASS |
| `Matrix-Game-4.0` | 2-Host 16-TC Host-SP DP (DP=2, U=8, R=1) | 2 (`16`) | `2×8×1` | `720p` | 4 | 106.21 | 75.36 | 35.1 | PASS | PASS |
| `Matrix-Game-4.0` | 4-Host 32-TC Intra-Chip DP (DP=16, U=2, R=1) | 4 (`32`) | `16×2×1` | `480p` | 4 | 81.39 | 98.29 | 309.5 | PASS | PASS |
| `Matrix-Game-4.0` | 4-Host 32-TC Intra-Chip DP (DP=16, U=2, R=1) | 4 (`32`) | `16×2×1` | `720p` | 4 | 170.42 | 46.95 | 147.1 | PASS | PASS |
| `Matrix-Game-4.0` | 4-Host 32-TC SubHost DP (DP=8, U=4, R=1) | 4 (`32`) | `8×4×1` | `480p` | 4 | 69.87 | 114.50 | 168.6 | PASS | PASS |
| `Matrix-Game-4.0` | 4-Host 32-TC SubHost DP (DP=8, U=4, R=1) | 4 (`32`) | `8×4×1` | `720p` | 4 | 121.71 | 65.73 | 106.4 | PASS | PASS |
| `Matrix-Game-4.0` | 4-Host 32-TC Host-Ulysses DP (DP=4, U=8, R=1) | 4 (`32`) | `4×8×1` | `480p` | 4 | 66.58 | 120.16 | 105.8 | PASS | PASS |
| `Matrix-Game-4.0` | 4-Host 32-TC Host-Ulysses DP (DP=4, U=8, R=1) | 4 (`32`) | `4×8×1` | `720p` | 4 | 104.11 | 76.84 | 65.1 | PASS | PASS |
| `Matrix-Game-4.0` | 4-Host 32-TC Hybrid U4xR2 DP (DP=4, U=4, R=2) | 4 (`32`) | `4×4×2` | `480p` | 4 | 59.31 | 134.87 | 110.2 | PASS | PASS |
| `Matrix-Game-4.0` | 4-Host 32-TC Hybrid U4xR2 DP (DP=4, U=4, R=2) | 4 (`32`) | `4×4×2` | `720p` | 4 | 82.98 | 96.40 | 64.6 | PASS | PASS |
| `Matrix-Game-2.0` | 1-Host 8-TC Intra-Chip DP (DP=4, U=2, R=1) | 1 (`8`) | `4×2×1` | `480p` | 4 | 52.81 | 151.49 | 108.8 | PASS | PASS |
| `Matrix-Game-2.0` | 1-Host 8-TC Intra-Chip DP (DP=4, U=2, R=1) | 1 (`8`) | `4×2×1` | `720p` | 4 | 95.25 | 84.00 | 63.4 | PASS | PASS |
| `Matrix-Game-2.0` | 1-Host 8-TC SubHost DP (DP=2, U=4, R=1) | 1 (`8`) | `2×4×1` | `480p` | 4 | 49.11 | 163.13 | 55.2 | PASS | PASS |
| `Matrix-Game-2.0` | 1-Host 8-TC SubHost DP (DP=2, U=4, R=1) | 1 (`8`) | `2×4×1` | `720p` | 4 | 70.50 | 113.49 | 34.6 | PASS | PASS |
| `Matrix-Game-2.0` | 1-Host 8-TC Full-Host Ulysses (DP=1, U=8, R=1) | 1 (`8`) | `1×8×1` | `480p` | 4 | 51.40 | 155.64 | 29.2 | PASS | PASS |
| `Matrix-Game-2.0` | 1-Host 8-TC Full-Host Ulysses (DP=1, U=8, R=1) | 1 (`8`) | `1×8×1` | `720p` | 4 | 67.05 | 119.31 | 17.8 | PASS | PASS |
| `Matrix-Game-2.0` | 1-Host 8-TC Hybrid U4xR2 (DP=1, U=4, R=2) | 1 (`8`) | `1×4×2` | `480p` | 4 | 49.71 | 160.95 | 28.9 | PASS | PASS |
| `Matrix-Game-2.0` | 1-Host 8-TC Hybrid U4xR2 (DP=1, U=4, R=2) | 1 (`8`) | `1×4×2` | `720p` | 4 | 57.56 | 138.98 | 17.9 | PASS | PASS |
| `Matrix-Game-2.0` | 2-Host 16-TC Intra-Chip DP (DP=8, U=2, R=1) | 2 (`16`) | `8×2×1` | `480p` | 4 | 53.15 | 150.52 | 192.1 | PASS | PASS |
| `Matrix-Game-2.0` | 2-Host 16-TC Intra-Chip DP (DP=8, U=2, R=1) | 2 (`16`) | `8×2×1` | `720p` | 4 | 94.83 | 84.36 | 108.1 | PASS | PASS |
| `Matrix-Game-2.0` | 2-Host 16-TC SubHost DP (DP=4, U=4, R=1) | 2 (`16`) | `4×4×1` | `480p` | 4 | 53.99 | 148.17 | 107.0 | PASS | PASS |
| `Matrix-Game-2.0` | 2-Host 16-TC SubHost DP (DP=4, U=4, R=1) | 2 (`16`) | `4×4×1` | `720p` | 4 | 73.84 | 108.34 | 64.2 | PASS | PASS |
| `Matrix-Game-2.0` | 2-Host 16-TC Host-SP DP (DP=2, U=8, R=1) | 2 (`16`) | `2×8×1` | `480p` | 4 | 51.09 | 156.59 | 55.5 | PASS | PASS |
| `Matrix-Game-2.0` | 2-Host 16-TC Host-SP DP (DP=2, U=8, R=1) | 2 (`16`) | `2×8×1` | `720p` | 4 | 65.66 | 121.85 | 33.6 | PASS | PASS |
| `Matrix-Game-2.0` | 4-Host 32-TC Intra-Chip DP (DP=16, U=2, R=1) | 4 (`32`) | `16×2×1` | `480p` | 4 | 54.01 | 148.12 | 289.0 | PASS | PASS |
| `Matrix-Game-2.0` | 4-Host 32-TC Intra-Chip DP (DP=16, U=2, R=1) | 4 (`32`) | `16×2×1` | `720p` | 4 | 95.21 | 84.02 | 153.0 | PASS | PASS |
| `Matrix-Game-2.0` | 4-Host 32-TC SubHost DP (DP=8, U=4, R=1) | 4 (`32`) | `8×4×1` | `480p` | 4 | 52.91 | 151.21 | 168.6 | PASS | PASS |
| `Matrix-Game-2.0` | 4-Host 32-TC SubHost DP (DP=8, U=4, R=1) | 4 (`32`) | `8×4×1` | `720p` | 4 | 74.46 | 107.44 | 104.6 | PASS | PASS |
| `Matrix-Game-2.0` | 4-Host 32-TC Host-Ulysses DP (DP=4, U=8, R=1) | 4 (`32`) | `4×8×1` | `480p` | 4 | 54.38 | 147.13 | 106.8 | PASS | PASS |
| `Matrix-Game-2.0` | 4-Host 32-TC Host-Ulysses DP (DP=4, U=8, R=1) | 4 (`32`) | `4×8×1` | `720p` | 4 | 67.54 | 118.45 | 60.5 | PASS | PASS |
| `Matrix-Game-2.0` | 4-Host 32-TC Hybrid U4xR2 DP (DP=4, U=4, R=2) | 4 (`32`) | `4×4×2` | `480p` | 4 | 48.86 | 163.72 | 108.9 | PASS | PASS |
| `Matrix-Game-2.0` | 4-Host 32-TC Hybrid U4xR2 DP (DP=4, U=4, R=2) | 4 (`32`) | `4×4×2` | `720p` | 4 | 57.37 | 139.45 | 66.7 | PASS | PASS |
| `HunyuanWorld-Voyager` | 1-Host 8-TC Intra-Chip DP (DP=4, U=2, R=1) | 1 (`8`) | `4×2×1` | `480p` | 8 | 394.85 | 20.26 | 105.0 | PASS | MISS |
| `HunyuanWorld-Voyager` | 1-Host 8-TC Intra-Chip DP (DP=4, U=2, R=1) | 1 (`8`) | `4×2×1` | `720p` | 8 | 1109.63 | 7.21 | 55.3 | MISS | MISS |
| `HunyuanWorld-Voyager` | 1-Host 8-TC SubHost DP (DP=2, U=4, R=1) | 1 (`8`) | `2×4×1` | `480p` | 8 | 360.75 | 22.18 | 56.4 | PASS | MISS |
| `HunyuanWorld-Voyager` | 1-Host 8-TC SubHost DP (DP=2, U=4, R=1) | 1 (`8`) | `2×4×1` | `720p` | 8 | 650.91 | 12.29 | 34.0 | MISS | MISS |
| `HunyuanWorld-Voyager` | 1-Host 8-TC Full-Host Ulysses (DP=1, U=8, R=1) | 1 (`8`) | `1×8×1` | `480p` | 8 | 336.21 | 23.79 | 28.6 | PASS | MISS |
| `HunyuanWorld-Voyager` | 1-Host 8-TC Full-Host Ulysses (DP=1, U=8, R=1) | 1 (`8`) | `1×8×1` | `720p` | 8 | 517.76 | 15.45 | 17.7 | MISS | MISS |
| `HunyuanWorld-Voyager` | 1-Host 8-TC Hybrid U4xR2 (DP=1, U=4, R=2) | 1 (`8`) | `1×4×2` | `480p` | 8 | 264.43 | 30.25 | 28.3 | PASS | PASS |
| `HunyuanWorld-Voyager` | 1-Host 8-TC Hybrid U4xR2 (DP=1, U=4, R=2) | 1 (`8`) | `1×4×2` | `720p` | 8 | 393.04 | 20.35 | 18.1 | PASS | MISS |
| `HunyuanWorld-Voyager` | 2-Host 16-TC Intra-Chip DP (DP=8, U=2, R=1) | 2 (`16`) | `8×2×1` | `480p` | 8 | 378.27 | 21.15 | 176.4 | PASS | MISS |
| `HunyuanWorld-Voyager` | 2-Host 16-TC Intra-Chip DP (DP=8, U=2, R=1) | 2 (`16`) | `8×2×1` | `720p` | 8 | 1091.48 | 7.33 | 92.4 | MISS | MISS |
| `HunyuanWorld-Voyager` | 2-Host 16-TC SubHost DP (DP=4, U=4, R=1) | 2 (`16`) | `4×4×1` | `480p` | 8 | 358.52 | 22.31 | 107.7 | PASS | MISS |
| `HunyuanWorld-Voyager` | 2-Host 16-TC SubHost DP (DP=4, U=4, R=1) | 2 (`16`) | `4×4×1` | `720p` | 8 | 663.99 | 12.05 | 65.9 | MISS | MISS |
| `HunyuanWorld-Voyager` | 2-Host 16-TC Host-SP DP (DP=2, U=8, R=1) | 2 (`16`) | `2×8×1` | `480p` | 8 | 314.83 | 25.41 | 56.0 | PASS | PASS |
| `HunyuanWorld-Voyager` | 2-Host 16-TC Host-SP DP (DP=2, U=8, R=1) | 2 (`16`) | `2×8×1` | `720p` | 8 | 504.92 | 15.84 | 34.6 | MISS | MISS |
| `HunyuanWorld-Voyager` | 4-Host 32-TC Intra-Chip DP (DP=16, U=2, R=1) | 4 (`32`) | `16×2×1` | `480p` | 8 | 395.42 | 20.23 | 284.1 | PASS | MISS |
| `HunyuanWorld-Voyager` | 4-Host 32-TC Intra-Chip DP (DP=16, U=2, R=1) | 4 (`32`) | `16×2×1` | `720p` | 8 | 1088.51 | 7.35 | 132.4 | MISS | MISS |
| `HunyuanWorld-Voyager` | 4-Host 32-TC SubHost DP (DP=8, U=4, R=1) | 4 (`32`) | `8×4×1` | `480p` | 8 | 360.98 | 22.16 | 172.0 | PASS | MISS |
| `HunyuanWorld-Voyager` | 4-Host 32-TC SubHost DP (DP=8, U=4, R=1) | 4 (`32`) | `8×4×1` | `720p` | 8 | 653.16 | 12.25 | 103.8 | MISS | MISS |
| `HunyuanWorld-Voyager` | 4-Host 32-TC Host-Ulysses DP (DP=4, U=8, R=1) | 4 (`32`) | `4×8×1` | `480p` | 8 | 323.50 | 24.73 | 108.8 | PASS | PASS |
| `HunyuanWorld-Voyager` | 4-Host 32-TC Host-Ulysses DP (DP=4, U=8, R=1) | 4 (`32`) | `4×8×1` | `720p` | 8 | 513.95 | 15.56 | 64.0 | MISS | MISS |
| `HunyuanWorld-Voyager` | 4-Host 32-TC Hybrid U4xR2 DP (DP=4, U=4, R=2) | 4 (`32`) | `4×4×2` | `480p` | 8 | 289.56 | 27.62 | 106.2 | PASS | PASS |
| `HunyuanWorld-Voyager` | 4-Host 32-TC Hybrid U4xR2 DP (DP=4, U=4, R=2) | 4 (`32`) | `4×4×2` | `720p` | 8 | 401.98 | 19.90 | 63.1 | PASS | MISS |

---

## 6. Suite 3: Multi-Host Concurrent Session Dispatch & Async CPU Video Encode Sweep (48 Cases, `32` TensorCores)

Raw JSON: [`results/concurrent_player_sessions_sla.json`](./results/concurrent_player_sessions_sla.json). *(See Section 1 Caveat 2 above: `chunk_to_chunk_p50_ms` and `sustained_player_fps_p50` below reflect single-sample 6-layer RPC RTT + async `video-encode-svc` load rather than full-depth `n_layers × denoise_steps` latency.)*

| Model | Res | Sharding (`DP × U`) | Concurrent Sessions | Sample RPC `p50` (ms) | Sample RPC `p95` (ms) | Sample RPC FPS | Slice Aggregate FPS (w/ Async MP4) | Async MP4 Encode (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Matrix-Game-4.0` | `480p` | `16×2` | 1 | 6.93 | 6.99 | 1153.84 | 70.1 | 108.10 |
| `Matrix-Game-4.0` | `480p` | `16×2` | 2 | 10.72 | 11.34 | 807.72 | 145.6 | 103.55 |
| `Matrix-Game-4.0` | `480p` | `16×2` | 4 | 10.18 | 14.21 | 772.02 | 227.7 | 126.62 |
| `Matrix-Game-4.0` | `480p` | `16×2` | 8 | 10.43 | 14.65 | 799.17 | 387.4 | 153.16 |
| `Matrix-Game-4.0` | `480p` | `16×2` | 16 | 10.91 | 27.74 | 692.62 | 906.6 | 217.16 |
| `Matrix-Game-4.0` | `480p` | `16×2` | 32 | 13.25 | 26.61 | 579.36 | 1344.8 | 283.67 |
| `Matrix-Game-4.0` | `720p` | `8×4` | 1 | 10.17 | 13.82 | 744.30 | 51.6 | 146.87 |
| `Matrix-Game-4.0` | `720p` | `8×4` | 2 | 12.79 | 18.00 | 613.67 | 104.8 | 143.39 |
| `Matrix-Game-4.0` | `720p` | `8×4` | 4 | 12.01 | 16.12 | 690.23 | 150.8 | 199.78 |
| `Matrix-Game-4.0` | `720p` | `8×4` | 8 | 12.93 | 20.60 | 619.27 | 195.5 | 289.30 |
| `Matrix-Game-4.0` | `720p` | `8×4` | 16 | 16.89 | 39.22 | 474.98 | 428.4 | 464.74 |
| `Matrix-Game-4.0` | `720p` | `8×4` | 32 | 23.56 | 53.40 | 320.34 | 786.7 | 495.28 |
| `Matrix-Game-2.0` | `480p` | `16×2` | 1 | 5.67 | 5.89 | 1440.90 | 71.2 | 107.69 |
| `Matrix-Game-2.0` | `480p` | `16×2` | 2 | 9.50 | 10.45 | 917.80 | 129.2 | 119.50 |
| `Matrix-Game-2.0` | `480p` | `16×2` | 4 | 9.38 | 11.57 | 886.17 | 207.8 | 136.70 |
| `Matrix-Game-2.0` | `480p` | `16×2` | 8 | 9.11 | 15.12 | 866.50 | 361.2 | 161.29 |
| `Matrix-Game-2.0` | `480p` | `16×2` | 16 | 10.67 | 21.62 | 690.61 | 763.6 | 256.49 |
| `Matrix-Game-2.0` | `480p` | `16×2` | 32 | 12.35 | 23.69 | 639.43 | 1807.1 | 212.67 |
| `Matrix-Game-2.0` | `720p` | `16×2` | 1 | 7.61 | 7.77 | 1051.00 | 51.3 | 147.25 |
| `Matrix-Game-2.0` | `720p` | `16×2` | 2 | 7.52 | 17.39 | 823.78 | 77.2 | 197.63 |
| `Matrix-Game-2.0` | `720p` | `16×2` | 4 | 11.30 | 16.25 | 680.87 | 128.5 | 216.58 |
| `Matrix-Game-2.0` | `720p` | `16×2` | 8 | 10.25 | 20.33 | 727.68 | 174.1 | 339.21 |
| `Matrix-Game-2.0` | `720p` | `16×2` | 16 | 15.77 | 43.97 | 502.14 | 408.6 | 486.18 |
| `Matrix-Game-2.0` | `720p` | `16×2` | 32 | 16.20 | 44.36 | 490.07 | 801.4 | 487.00 |
| `HunyuanWorld-Voyager` | `480p` | `8×4` | 1 | 8.49 | 8.71 | 943.38 | 70.6 | 108.49 |
| `HunyuanWorld-Voyager` | `480p` | `8×4` | 2 | 11.16 | 12.52 | 753.78 | 122.2 | 125.68 |
| `HunyuanWorld-Voyager` | `480p` | `8×4` | 4 | 10.75 | 13.40 | 754.17 | 240.3 | 127.13 |
| `HunyuanWorld-Voyager` | `480p` | `8×4` | 8 | 10.93 | 18.82 | 728.94 | 346.6 | 166.48 |
| `HunyuanWorld-Voyager` | `480p` | `8×4` | 16 | 13.43 | 32.08 | 539.66 | 786.8 | 247.66 |
| `HunyuanWorld-Voyager` | `480p` | `8×4` | 32 | 20.46 | 32.86 | 387.34 | 1877.1 | 208.29 |
| `HunyuanWorld-Voyager` | `720p` | `8×4` | 1 | 11.12 | 11.91 | 712.02 | 50.9 | 148.21 |
| `HunyuanWorld-Voyager` | `720p` | `8×4` | 2 | 14.51 | 18.30 | 565.88 | 78.2 | 196.41 |
| `HunyuanWorld-Voyager` | `720p` | `8×4` | 4 | 13.45 | 26.88 | 530.03 | 122.3 | 223.53 |
| `HunyuanWorld-Voyager` | `720p` | `8×4` | 8 | 14.04 | 34.41 | 541.01 | 194.0 | 287.94 |
| `HunyuanWorld-Voyager` | `720p` | `8×4` | 16 | 18.54 | 34.78 | 417.20 | 366.0 | 525.65 |
| `HunyuanWorld-Voyager` | `720p` | `8×4` | 32 | 26.79 | 70.92 | 275.54 | 730.2 | 532.31 |
| `HunyuanWorld-1` | `480p` | `8×4` | 1 | 6.94 | 7.04 | 1152.39 | 70.9 | 108.18 |
| `HunyuanWorld-1` | `480p` | `8×4` | 2 | 10.20 | 11.16 | 833.67 | 144.8 | 104.89 |
| `HunyuanWorld-1` | `480p` | `8×4` | 4 | 10.16 | 11.30 | 821.49 | 227.8 | 122.73 |
| `HunyuanWorld-1` | `480p` | `8×4` | 8 | 9.90 | 20.33 | 743.87 | 398.6 | 138.56 |
| `HunyuanWorld-1` | `480p` | `8×4` | 16 | 12.22 | 20.71 | 681.93 | 796.6 | 240.34 |
| `HunyuanWorld-1` | `480p` | `8×4` | 32 | 19.52 | 38.65 | 418.40 | 1826.5 | 206.27 |
| `HunyuanWorld-1` | `720p` | `8×4` | 1 | 8.39 | 8.73 | 946.15 | 50.2 | 149.78 |
| `HunyuanWorld-1` | `720p` | `8×4` | 2 | 12.00 | 15.39 | 682.76 | 104.3 | 145.24 |
| `HunyuanWorld-1` | `720p` | `8×4` | 4 | 11.62 | 26.91 | 601.25 | 147.1 | 200.07 |
| `HunyuanWorld-1` | `720p` | `8×4` | 8 | 12.66 | 25.10 | 609.94 | 172.3 | 339.53 |
| `HunyuanWorld-1` | `720p` | `8×4` | 16 | 17.47 | 42.83 | 443.90 | 394.1 | 497.53 |
| `HunyuanWorld-1` | `720p` | `8×4` | 32 | 24.13 | 64.81 | 303.55 | 807.2 | 482.42 |
