#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A zgokce
#SBATCH -J bench_eval_swin_tiny_asl_tr256_lr1e-4_te256
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00-04:00
#SBATCH --output=/arf/scratch/zgokce/logs/swin_tiny/slurm-%x-job%j-%t.out
#SBATCH --error=/arf/scratch/zgokce/logs/swin_tiny/slurm-%x-job%j-%t.err

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

CONFIG="./configs/SLR/bench_swin_tiny/eval/swin_tiny_aslcitizen100_from_train256_lr1e-4_test256.py"
TRAIN_WORKDIR="/arf/scratch/zgokce/workdir/swin_tiny/aslcitizen100/train_256_lr1e-4"
EVAL_DIR="/arf/scratch/zgokce/workdir/swin_tiny/aslcitizen100/eval_from_train256_lr1e-4_test256"
REPORT_DIR="/arf/scratch/zgokce/workdir/swin_tiny/aslcitizen100/reports"

mkdir -p "$EVAL_DIR" "$REPORT_DIR" /arf/scratch/zgokce/logs/swin_tiny

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
  --model swin_tiny \
  --dataset aslcitizen100 \
  --train_res 256_lr1e-4 \
  --test_type 256 \
  --config "$CONFIG" \
  --ckpt "$CKPT_PATH" \
  --eval_log "$EVAL_DIR/eval.log" \
  --report_dir "$REPORT_DIR"

exit
