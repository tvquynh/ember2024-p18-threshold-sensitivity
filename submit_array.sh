#!/bin/bash
# submit_array.sh — Launch the experiment across 10 SLURM nodes (1 seed per node).
#
# USAGE (run on the SLURM submit host):
#   bash submit_array.sh                  # full run
#   bash submit_array.sh --prototype      # smoke (~10 min on each node)
#   bash submit_array.sh --resume         # skip per-run files already done
#
# Maps the 10 fixed seeds (config.SEEDS) to 10 nodes via the NODES array below.
# Edit NODES (and the per-node CPU/RAM allocation) for your cluster topology.
# The first node receives seed 42 with a slightly larger CPU/RAM allocation
# because that workload is the heaviest in our setup.
#
# Output: ${OUT_DIR}/seed_<N>/...
# Aggregate after all 10 done with: python aggregate.py

set -euo pipefail

PAPER=p18_threshold_sensitivity
# Edit the four paths below to match your cluster's shared filesystem layout.
CODE_DIR=${CODE_DIR:-/srv/nfs/code/$PAPER}
OUT_DIR=${OUT_DIR:-/srv/nfs/results/$PAPER}
PARQUET_DIR=${PARQUET_DIR:?PARQUET_DIR must be set to the EMBER2024 parquet directory}
SBATCH_TEMPLATE=$CODE_DIR/run_seed.sbatch

# Parse flags
PROTOTYPE_FLAG=""
RESUME_FLAG=""
for arg in "$@"; do
    case $arg in
        --prototype) PROTOTYPE_FLAG="--prototype" ;;
        --resume)    RESUME_FLAG="--resume" ;;
        *)           echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# Sanity checks
[ -f "$SBATCH_TEMPLATE" ] || { echo "ERROR: $SBATCH_TEMPLATE not found"; exit 1; }
[ -d "$PARQUET_DIR" ]     || { echo "ERROR: $PARQUET_DIR not on this node"; exit 1; }
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Submitting P18 array: 10 seeds → 10 nodes"
echo "  CODE_DIR    : $CODE_DIR"
echo "  OUT_DIR     : $OUT_DIR"
echo "  PARQUET_DIR : $PARQUET_DIR"
echo "  PROTOTYPE   : ${PROTOTYPE_FLAG:-NO (full run)}"
echo "  RESUME      : ${RESUME_FLAG:-NO}"
echo "============================================================"

# Seed-to-node mapping (uses 10 fixed seeds from config.SEEDS).
# Edit the NODES array to match the available nodes on your cluster.
SEEDS=(42 123 456 789 1011 2026 3141 4242 5555 6789)
NODES=(node01 node02 node03 node04 node05 node06 node07 node08 node09 node10)

# Tune the first node (heaviest workload, seed 42) and the rest separately.
for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    node=${NODES[$i]}

    if [ "$i" = "0" ]; then
        CPUS=120
        MEM=230G
    else
        CPUS=56
        MEM=240G
    fi

    job_name=${PAPER}_seed_${seed}
    sbatch \
        --nodelist=$node \
        --cpus-per-task=$CPUS \
        --mem=$MEM \
        --job-name=$job_name \
        --output=$OUT_DIR/seed_${seed}.out \
        --error=$OUT_DIR/seed_${seed}.err \
        --export=ALL,SEED=$seed,CODE_DIR=$CODE_DIR,OUT_DIR=$OUT_DIR,PARQUET_DIR=$PARQUET_DIR,PROTOTYPE_FLAG="$PROTOTYPE_FLAG",RESUME_FLAG="$RESUME_FLAG" \
        $SBATCH_TEMPLATE

    echo "  → seed=$seed → $node ($CPUS cores, $MEM)"
done

echo "============================================================"
echo "All 10 jobs submitted. Monitor:"
echo "  squeue -u \$(whoami)"
echo "  tail -f $OUT_DIR/seed_42.out"
echo "  ls $OUT_DIR/seed_*/done.json   # marker files appear when each seed finishes"
echo ""
echo "When all 10 done.json present, aggregate:"
echo "  python $CODE_DIR/aggregate.py --seeds_dir $OUT_DIR --output_dir $OUT_DIR/aggregated"
echo "============================================================"
