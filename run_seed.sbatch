#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=14-00:00:00
# CPUs / mem / nodelist / output / error / job-name are passed at submit time;
# see submit_array.sh for the per-node tuning.

set -euo pipefail

# Activate the cluster-wide Python virtualenv. Adjust this path to your cluster.
source "${VENV_DIR:?VENV_DIR must point at the shared Python virtualenv}/bin/activate"

cd "$CODE_DIR"

# Print runtime context for debugging.
echo "============================================================"
echo "run_seed.sbatch"
echo "  hostname     : $(hostname)"
echo "  date         : $(date)"
echo "  SLURM_JOB_ID : ${SLURM_JOB_ID:-NA}"
echo "  CPUS         : ${SLURM_CPUS_PER_TASK:-NA}"
echo "  MEM          : ${SLURM_MEM_PER_NODE:-NA}"
echo "  SEED         : ${SEED}"
echo "  CODE_DIR     : ${CODE_DIR}"
echo "  OUT_DIR      : ${OUT_DIR}"
echo "  PARQUET_DIR  : ${PARQUET_DIR:?PARQUET_DIR must point at the EMBER2024 parquet directory}"
echo "============================================================"

python run_seed.py \
    --seed="$SEED" \
    --output_dir="$OUT_DIR/seed_$SEED" \
    --parquet_dir="$PARQUET_DIR" \
    --num_threads="$SLURM_CPUS_PER_TASK" \
    ${PROTOTYPE_FLAG:-} \
    ${RESUME_FLAG:-}

echo "DONE: seed=$SEED at $(date)"
