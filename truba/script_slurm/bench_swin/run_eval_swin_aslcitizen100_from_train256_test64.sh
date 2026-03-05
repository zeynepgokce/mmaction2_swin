#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A zgokce
#SBATCH -J bench_eval_swin_aslcitizen_tr256_te64
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00-04:00
#SBATCH --output=/arf/scratch/zgokce/logs/bench/swin/slurm-%x-job%j-%t.out
#SBATCH --error=/arf/scratch/zgokce/logs/bench/swin/slurm-%x-job%j-%t.err

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

CONFIG="./configs/SLR/bench_swin/eval/swin_aslcitizen100_from_train256_test64.py"
TRAIN_WORKDIR="/arf/scratch/zgokce/bench/swin/aslcitizen100/train_256"
EVAL_DIR="/arf/scratch/zgokce/bench/swin/aslcitizen100/eval_from_train256_test64"
REPORT_DIR="/arf/scratch/zgokce/bench/reports"

mkdir -p "$EVAL_DIR" "$REPORT_DIR" /arf/scratch/zgokce/logs/bench/swin

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
  --model swin \
  --dataset aslcitizen100 \
  --train_res 256 \
  --test_type 64 \
  --config "$CONFIG" \
  --ckpt "$CKPT_PATH" \
  --eval_log "$EVAL_DIR/eval.log" \
  --report_dir "$REPORT_DIR"

exit
