#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A zgokce
#SBATCH -J uniformerv2_base_wlasl100_64x64
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=01-00:00
#SBATCH --output=/arf/scratch/zgokce/logs/uniformerv2_base_wlasl100_64x64/slurm-%x-job%j-%t.out  # %x=job name, %j=job ID, %t=task ID
#SBATCH --error=/arf/scratch/zgokce/logs/uniformerv2_base_wlasl100_64x64/slurm-%x-job%j-%t.err

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



# variables to change
dataset_root="/arf/scratch/zgokce/data"
res="64"
model_name="uniformerv2"
model_type="base"
dataset="WLASL100"
EXP_NAME="base"
CONFIG="./configs/SLR/uniformerv2_wlasl100_64x64/uniformerv2_base_wlasl100_64x64_16f.py"
RUN_DIR="/arf/scratch/zgokce/workdir/${dataset}_${res}x${res}/${model_name}/${model_type}/${EXP_NAME}_job${SLURM_JOB_ID}"
dump="${RUN_DIR}/result.pkl"

echo "==> CONFIG     : $CONFIG"
echo "==> WORK DIR   : $RUN_DIR"
echo "==> DUMP FILE  : $dump"

mkdir -p "$RUN_DIR"
#train
srun python ./tools/train.py "$CONFIG" \
  --cfg-options work_dir="$RUN_DIR" dataset_root="$dataset_root" 2>&1 | tee -a "$RUN_DIR/train.log"

#TEST: find best checkpoint path in the workdir
CKPT_PATH="$(find "$RUN_DIR" -type f -name 'best_acc_top1_epoch*.pth' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
echo "==> CKPT_PATH     : $CKPT_PATH"

srun python ./tools/test.py "$CONFIG" "$CKPT_PATH" \
  --cfg-options work_dir="$RUN_DIR" dataset_root="$dataset_root" \
  --dump "$dump"

exit
