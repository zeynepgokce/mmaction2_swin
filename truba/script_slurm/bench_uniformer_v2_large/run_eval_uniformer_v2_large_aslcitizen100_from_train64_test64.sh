#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A zgokce
#SBATCH -J eval_univ2l_aslcitizen100_tr64_test64
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00-02:00
#SBATCH --output=/arf/scratch/zgokce/logs/uniformer_v2_large/slurm-%x-job%j-%t.out
#SBATCH --error=/arf/scratch/zgokce/logs/uniformer_v2_large/slurm-%x-job%j-%t.err

source /etc/profile.d/modules.sh
module purge
module load lib/cuda/11.8
module load miniconda3

echo "NODE: $(hostname)"
nvidia-smi

wdir=/arf/home/zgokce/code/mmaction2_swin
cd $wdir

export PYTHONPATH=/arf/home/zgokce/miniconda3/envs/open-mmlab/lib/python3.7/site-packages
conda activate open-mmlab

CONFIG="./configs/SLR/bench_uniformer_v2_large/eval/uniformer_v2_large_aslcitizen100_from_train64_test64.py"
TRAIN_WORKDIR="/arf/scratch/zgokce/workdir/uniformer_v2_large/aslcitizen100/train_64"
EVAL_DIR="/arf/scratch/zgokce/workdir/uniformer_v2_large/aslcitizen100/eval_from_train64_test64"
REPORT_DIR="/arf/scratch/zgokce/workdir/uniformer_v2_large/aslcitizen100/reports"

mkdir -p "$EVAL_DIR" "$REPORT_DIR" /arf/scratch/zgokce/logs/uniformer_v2_large

CKPT_PATH="$(find "$TRAIN_WORKDIR" -type f -name 'best_acc_top1_epoch*.pth' \
  -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
if [ -z "$CKPT_PATH" ]; then
  echo "ERROR: No best checkpoint found in $TRAIN_WORKDIR" >&2; exit 1
fi
echo "==> CKPT: $CKPT_PATH"

srun python ./tools/test.py "$CONFIG" "$CKPT_PATH" \
  --cfg-options work_dir="$EVAL_DIR" \
  2>&1 | tee -a "$EVAL_DIR/eval.log"

srun python ./tools/bench_report.py \
  --model uniformer_v2_large \
  --dataset aslcitizen100 \
  --train_res 64 \
  --test_type 64 \
  --config "$CONFIG" \
  --ckpt "$CKPT_PATH" \
  --eval_log "$EVAL_DIR/eval.log" \
  --report_dir "$REPORT_DIR"

exit
