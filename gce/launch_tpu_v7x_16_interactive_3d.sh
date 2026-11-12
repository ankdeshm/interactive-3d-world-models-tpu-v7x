#!/usr/bin/env bash
# Launch 4-Host Cloud TPU v7x-16 (32 TensorCores, 2x2x4) Interactive 3D & Game World Models Benchmark
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-YOUR_GCP_PROJECT_ID}"
ZONE="${ZONE:-us-central1-c}"
TPU_NAME="${TPU_NAME:-interactive-3d-v7x-16-slice}"
GCS_MODELS_URI="${GCS_MODELS_URI:-gs://YOUR_GCS_BUCKET/models}"

echo "[1/3] Staging Matrix-Game-4.0, Matrix-Game-2.0, HunyuanWorld-Voyager, HunyuanWorld-1 into /dev/shm..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="python3 ~/interactive-3d-world-models-tpu-v7x/scripts/gcs_stager.py --gcs-prefix=${GCS_MODELS_URI} --shm-dir=/dev/shm/world3d_models"

echo "[2/3] Starting persistent TPU host-agents on Workers 0..3 (port 19110)..."
for WORKER in 0 1 2 3; do
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --worker="${WORKER}" \
    --command="nohup python3 ~/interactive-3d-world-models-tpu-v7x/scripts/run_interactive_3d_worlds_v7x.py --mode=host-agent --host-rank=${WORKER} --agent-port=19110 --shm-root=/dev/shm/world3d_models > /tmp/world3d_agent_${WORKER}.log 2>&1 &"
done

sleep 8
WORKER_IPS=$(gcloud compute tpus tpu-vm describe "${TPU_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --format="value(networkEndpoints[].ipAddress)" | tr ';' '\n' | sed 's|^|http://|;s|$|:19110|' | paste -sd, -)

echo "[3/3] Running causal streaming KV-cache, sharding, and CON=1..32 SLA sweeps from Worker 0..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --worker=0 \
  --command="python3 ~/interactive-3d-world-models-tpu-v7x/scripts/run_interactive_3d_worlds_v7x.py --mode=orchestrator --host-endpoints=${WORKER_IPS} --shm-root=/dev/shm/world3d_models --out-dir=~/interactive-3d-world-models-tpu-v7x/results"
