#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A zgokce
#SBATCH -J uniformerv2_base_wlasl100_64x64_exp_lr1e-05_c8x1_bs1
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02-00:00
#SBATCH --output=/arf/scratch/zgokce/logs/slurm-%x-job%j-%t.out  # %x=job name, %j=job ID, %t=task ID
#SBATCH --error=/arf/scratch/zgokce/logs/slurm-%x-job%j-%t.err

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
EXP_NAME="exp_lr1e-05_c8x1_bs1"
CONFIG="./configs/SLR/uniformerv2_wlasl100_64x64/finetune_WLASL100_uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb.py"
RUN_DIR="/arf/scratch/zgokce/workdir/${dataset}_${res}x${res}/${model_name}/${model_type}/${EXP_NAME}_job${SLURM_JOB_ID}"
CFG_OPTS="optim_wrapper.optimizer.lr=1e-05 train_pipeline.0.clip_len=8 val_pipeline.0.clip_len=8 test_pipeline.0.clip_len=8 train_dataloader.batch_size=1 val_dataloader.batch_size=1"
dump="${$RUN_DIR}/result.pkl"

echo "==> CONFIG     : $CONFIG"
echo "==> WORK DIR   : $RUN_DIR"

mkdir -p "$RUN_DIR"
#train
srun python ./tools/train.py "$CONFIG" \
  --cfg-options work_dir="$RUN_DIR" dataset_root="$dataset_root" $CFG_OPTS 2>&1 | tee -a "$RUN_DIR/train.log"

#TEST: find best checkpoint path in the workdir
CKPT_PATH="$(find "$RUN_DIR" -type f -name 'best_acc_top1_epoch*.pth' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"

srun python ./tools/test.py "$CONFIG" "$CKPT_PATH" \
  --cfg-options work_dir="$RUN_DIR" dataset_root="$dataset_root" $CFG_OPTS \
  --dump "$dump"

exit
