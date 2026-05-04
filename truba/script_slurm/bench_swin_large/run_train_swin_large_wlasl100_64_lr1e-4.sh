#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A zgokce
#SBATCH -J bench_train_swin_large_wlasl100_64_lr1e-4
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02-00:00
#SBATCH --output=/arf/scratch/zgokce/logs/swin_large/slurm-%x-job%j-%t.out
#SBATCH --error=/arf/scratch/zgokce/logs/swin_large/slurm-%x-job%j-%t.err

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

CONFIG="./configs/SLR/bench_swin_large/swin_large_wlasl100_train64_lr1e-4.py"
RUN_DIR="/arf/scratch/zgokce/workdir/swin_large/wlasl100/train_64_lr1e-4"
REPORT_DIR="/arf/scratch/zgokce/workdir/swin_large/wlasl100/reports"

echo "==> CONFIG   : $CONFIG"
echo "==> WORK DIR : $RUN_DIR"

mkdir -p "$RUN_DIR" "$REPORT_DIR" /arf/scratch/zgokce/logs/swin_large

srun python ./tools/train.py "$CONFIG" \
  --cfg-options work_dir="$RUN_DIR" \
  2>&1 | tee -a "$RUN_DIR/train.log"

CKPT_PATH="$(find "$RUN_DIR" -type f -name 'best_acc_top1_epoch*.pth' \
  -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
echo "==> BEST CKPT: $CKPT_PATH"

EVAL_DIR="${RUN_DIR}/eval_from_train64_test64"
mkdir -p "$EVAL_DIR"

srun python ./tools/test.py "$CONFIG" "$CKPT_PATH" \
  --cfg-options work_dir="$EVAL_DIR" \
  2>&1 | tee -a "$EVAL_DIR/eval.log"

srun python ./tools/bench_report.py \
  --model swin_large \
  --dataset wlasl100 \
  --train_res 64_lr1e-4 \
  --test_type 64 \
  --config "$CONFIG" \
  --ckpt "$CKPT_PATH" \
  --eval_log "$EVAL_DIR/eval.log" \
  --report_dir "$REPORT_DIR" \
  --train_log "$RUN_DIR/train.log"

exit
