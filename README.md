# Interactive 3D and Action-Conditioned World Models on Cloud TPU v7x

Causal rolling KV-cache streaming, multi-host sharding (`DP × Ulysses × Ring`), and CPU video-encoding offload benchmarks for four interactive 3D and gaming world models on a 16-chip Google Cloud TPU `v7x-16` slice (`32` TensorCores).

**Summary:** Combining causal rolling KV-cache reuse with `torch.compile` and `fp8_e4m3fn` MXU matmuls on TPU `v7x` reduces 8-frame `480p` interactive chunk latency to **30.02 ms (266.52 FPS) on `Skywork/Matrix-Game-2.0` (6.30× faster than eager `bf16` full-window recomputation at 189.22 ms)** and **52.69 ms (151.83 FPS) on `Skywork/Matrix-Game-4.0` (6.03× faster than eager `bf16` full-window at 317.86 ms)**. On the heavier 60-layer `tencent/HunyuanWorld-Voyager` (`13B`, `8` steps), the same optimizations reach **295.71 ms (27.05 FPS, 3.74× faster than eager `bf16` full-window)** at `480p`, while `720p` 8-step streaming peaks at **20.35 FPS (`393.04 ms`, meeting `16 FPS` but missing `24 FPS` without dropping to `4` steps)**.

---

## 1. Model, Framework, and Hardware

- **Models Evaluated** (checkpoint metadata in [`results/checkpoint_manifests.json`](./results/checkpoint_manifests.json)):
  - **`Skywork/Matrix-Game-4.0` (`~5.0B` params/scene, `28.13 GiB` `bfloat16` repo across 3 scenes + `taew2_2` VAE; `10.2 GiB` active scene):** 30-layer `CausalWanModel` DiT (`d = 3072`, `24` heads × `128`, `ffn = 14,336`) with action cross-attention (`4` denoise steps, `64`-frame rolling window).
  - **`Skywork/Matrix-Game-2.0` (`~1.8B` params, `19.97 GiB` `float32`/`bfloat16` repo; `6.8 GiB` active):** 30-layer distilled `CausalWanModel` DiT (`d = 1536`, `12` heads × `128`, `ffn = 8,960`) with keyboard/mouse cross-attention (`4` denoise steps, `64`-frame window).
  - **`tencent/HunyuanWorld-Voyager` (`~13.0B` params, `76.96 GiB` `float32`/`float16` checkpoint; `26.0 GiB` active shard group):** 60-layer RGB-D camera-pose & 3D point-cloud DiT (`d = 3072`, `24` heads × `128`, `ffn = 12,288`) + `AutoencoderKLCausal3D` (`8` default steps, `49`-frame window).
  - **`tencent/HunyuanWorld-1` (`1.45 GiB` `4×` rank-128 PanoDiT/PanoInpaint `bfloat16` adapters on a 57-layer `d = 3072` panoramic DiT):** 360-degree panoramic & layered 3D world generator (`8` default steps, `16`-frame window).
- **Framework & Measurement Scope:** Python 3.12, PyTorch + `torch_tpu` (`tpu` distributed ICI backend, `torch.compile(..., dynamic=False)`, `torch._scaled_mm` `float8_e4m3fn` with `bfloat16` output), and HTTP/1.1 Keep-Alive `ffmpeg` `libx264` chunk encoding ([`scripts/video_encode_server.py`](./scripts/video_encode_server.py)).
  - *Scope note:* [`scripts/run_interactive_3d_worlds_v7x.py`](./scripts/run_interactive_3d_worlds_v7x.py) loads the checkpoint tensors onto TPU HBM3e and times `6` architecturally matched DiT layers (scaled to full `30`–`60` layer depth × `num_steps`), a compiled 2-stage pixel-shuffle VAE decode pass, and hardware-calibrated 8-TensorCore ICI `all_reduce` collectives (see [`APPENDIX.md`](./APPENDIX.md) for methodology details and caveats).
- **Hardware:** Google Cloud TPU `v7x-16` slice (`2x2x4` 3D torus: `4` hosts × `4` physical chips = `16` chips / `32` TensorCores, `192 GiB` HBM3e per chip / `96 GiB` per TensorCore = `3,072 GiB` total HBM3e) paired with `c3-standard-44` (`Intel Xeon Platinum 8481C`, `44` vCPUs) CPU video-encoding nodes.

---

## 2. Main Results

### A. Causal Rolling KV-Cache vs. Full-Window Recomputation (`1` Host Sub-Slice `U=4`, `480p`, `8`-Frame Chunks)

Measured in [`results/causal_kv_vs_fullwindow_ablation.json`](./results/causal_kv_vs_fullwindow_ablation.json):

| Model (`Steps, Window`) | Eager `bf16` Full-Window | Eager `bf16` KV-Cache | Compiled `bf16` KV-Cache | Compiled `fp8` Full-Window | Compiled `fp8` KV-Cache | Comparison (`Compiled fp8 KV` vs. Baseline) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `Matrix-Game-2.0` (`4s, 64f`) | `189.22 ms` (`42.28` FPS) | `145.72 ms` (`54.90` FPS) | `33.12 ms` (`241.55` FPS) | `86.78 ms` (`92.19` FPS) | **`30.02 ms` (`266.52` FPS)** | **6.30× faster** than eager `bf16` full-window; **2.89×** vs. compiled `fp8` full-window |
| `Matrix-Game-4.0` (`4s, 64f`) | `317.86 ms` (`25.17` FPS) | `173.00 ms` (`46.24` FPS) | `61.90 ms` (`129.23` FPS) | `184.75 ms` (`43.30` FPS) | **`52.69 ms` (`151.83` FPS)** | **6.03× faster** than eager `bf16` full-window; **3.51×** vs. compiled `fp8` full-window |
| `HunyuanWorld-1` (`8s, 16f`) | `488.13 ms` (`16.39` FPS) | `427.02 ms` (`18.73` FPS) | `241.56 ms` (`33.12` FPS) | `297.57 ms` (`26.88` FPS) | **`216.73 ms` (`36.91` FPS)** | **2.25× faster** than eager `bf16` full-window; **1.37×** vs. compiled `fp8` full-window |
| `HunyuanWorld-Voyager` (`8s, 49f`) | `1,107.32 ms` (`7.22` FPS) | `598.26 ms` (`13.37` FPS) | `334.37 ms` (`23.93` FPS) | `788.73 ms` (`10.14` FPS) | **`295.71 ms` (`27.05` FPS)** | **3.74× faster** than eager `bf16` full-window; **2.67×** vs. compiled `fp8` full-window |

*Notes:* Uncompiled eager `fp8` regressed relative to eager `bf16` on `Matrix-Game-2.0` (`211.35` vs. `189.22 ms`) and `HunyuanWorld-1` (`579.62` vs. `488.13 ms`) due to unfused scale casts; `torch.compile` fuses the casts and also lowers 8-frame VAE decode latency from `3.44–54.29 ms` to `0.34–0.45 ms`.

### B. Full-Slice Multi-Host Sharding (`v7x-16`: `4` Hosts = `16` Chips / `32` TensorCores, Compiled `FP8` + KV-Cache)

Representative best-throughput (`DP`-heavy) and lowest-latency (`Ulysses × Ring`) 32-TensorCore configurations from [`results/sharding_topology_scaling.json`](./results/sharding_topology_scaling.json) (`8`-frame chunks; `Aggregate Slice FPS` includes synchronous remote MP4 chunk encoding):

| Model (`Steps`) | Resolution | 32-TC Sharding (`DP × U × R`) | Median Chunk Latency | Per-Stream FPS | Aggregate Slice FPS (w/ MP4 Offload) | Frame-Rate Target Check |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `Matrix-Game-2.0` (`4s`) | `480p` (`832×480`) | `16 × 2 × 1` (`16` streams) | `54.01 ms` | **148.12 FPS** | **289.0 FPS** | Passes `24 FPS` (`48.86 ms` / `163.72 FPS` at `4×4×2`) |
| `Matrix-Game-2.0` (`4s`) | `720p` (`1280×720`) | `16 × 2 × 1` (`16` streams) | `95.21 ms` | **84.02 FPS** | **153.0 FPS** | Passes `24 FPS` (`57.37 ms` / `139.45 FPS` at `4×4×2`) |
| `Matrix-Game-4.0` (`4s`) | `480p` (`832×480`) | `16 × 2 × 1` (`16` streams) | `81.39 ms` | **98.29 FPS** | **309.5 FPS** | Passes `24 FPS` (`59.31 ms` / `134.87 FPS` at `4×4×2`) |
| `Matrix-Game-4.0` (`4s`) | `720p` (`1280×720`) | `8 × 4 × 1` (`8` streams) | `121.71 ms` | **65.73 FPS** | **106.4 FPS** | Passes `24 FPS` (`82.98 ms` / `96.40 FPS` at `4×4×2`) |
| `HunyuanWorld-Voyager` (`8s`) | `480p` (`832×480`) | `4 × 4 × 2` (`4` streams) | `289.56 ms` | **27.62 FPS** | `106.2 FPS` (`284.1 FPS` at `16×2×1`) | Passes `24 FPS` at `4×4×2`; `20.23–22.16 FPS` (`16 FPS` only) at `U=2/4` |
| `HunyuanWorld-Voyager` (`8s`) | `720p` (`1280×720`) | `4 × 4 × 2` (`4` streams) | `401.98 ms` | **19.90 FPS** | `63.1 FPS` (`132.4 FPS` at `16×2×1`) | Passes `16 FPS` only at `U=4×R=2`; misses `24 FPS` at `8` steps (`26.37 FPS` at `4` steps) |

---

## 3. Reproducing the Main Results

Deploy the disaggregated `video-encode-svc` on CPU nodes and launch [`k8s/tpu-v7x-16-interactive-3d-worlds.yaml`](./k8s/tpu-v7x-16-interactive-3d-worlds.yaml), or run [`scripts/run_interactive_3d_worlds_v7x.py`](./scripts/run_interactive_3d_worlds_v7x.py) directly across the 4 TPU `v7x` hosts:

```bash
# 1. Stage checkpoints into /dev/shm and start host agents on TPU hosts 0..3
python3 scripts/run_interactive_3d_worlds_v7x.py --mode=stage-shm \
  --models-root=/models/models \
  --shm-root=/dev/shm/world3d_models

python3 scripts/run_interactive_3d_worlds_v7x.py --mode=host-agent \
  --host-rank="${HOST_RANK}" \
  --agent-port=19110 \
  --shm-root=/dev/shm/world3d_models &

# 2. Run Suite 1 (KV-cache/precision/compile), Suite 2 (1-4 host sharding), and Suite 3
python3 scripts/run_interactive_3d_worlds_v7x.py --mode=orchestrator \
  --config=configs/interactive_3d_worlds_v7x16.yaml \
  --host-endpoints=http://tpu-v7x-host-0:19110,http://tpu-v7x-host-1:19110,http://tpu-v7x-host-2:19110,http://tpu-v7x-host-3:19110 \
  --video-encode-url=http://video-encode-svc.default.svc.cluster.local:8090 \
  --shm-root=/dev/shm/world3d_models \
  --out-dir=./results
```

For GCE Cloud TPU VMs, configure `PROJECT_ID`, `ZONE`, `TPU_NAME`, and `GCS_MODELS_URI` and run [`gce/launch_tpu_v7x_16_interactive_3d.sh`](./gce/launch_tpu_v7x_16_interactive_3d.sh).

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Complete 266-case tables across Suite 1 (`152` precision/compile/chunk/step/resolution rows), Suite 2 (`66` 1-to-4 host sharding rows), and Suite 3 (`48` concurrent RPC/encoder rows), plus analysis of cases that regressed or missed `16/24 FPS` targets and Suite 3 measurement caveats.
- **[`results/`](./results/):** Raw JSON outputs (`causal_kv_vs_fullwindow_ablation.json`, `sharding_topology_scaling.json`, `concurrent_player_sessions_sla.json`, `checkpoint_manifests.json`, and `summary.json`).
