#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A zgokce
#SBATCH -J swin_small_wlasl_256x256
#SBATCH --gres=gpu:1                # Tek node'da 1 GPU
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=03-00:00
#SBATCH --output=logs/slurm-%x-%j-%t.out
#SBATCH --error=logs/slurm-%x-%j-%t.err
#SBATCH --array=0-3%4 # 4 deney, en fazla 4'ü aynı anda



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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ==== 4 deneyin tanımı ====
# 0: Base (override yok)
# 1: Temporal 16x1
# 2: Temporal 24x1
# 3: Temporal 32x1

CASE_TAG="base"
CFG_OPTS=""

case "$SLURM_ARRAY_TASK_ID" in
  0)
    CASE_TAG="base"
    CFG_OPTS=""  # override yok
    RUN_DIR="$work_dir/${CASE_TAG}_job3019268_${SLURM_ARRAY_TASK_ID}"
    ;;
  1)
    CASE_TAG="t16x1"
    CFG_OPTS="train_pipeline.0.clip_len=16 train_pipeline.0.frame_interval=1 \
      val_pipeline.0.clip_len=16   val_pipeline.0.frame_interval=1 \
      test_pipeline.0.clip_len=16  test_pipeline.0.frame_interval=1"
    RUN_DIR="$work_dir/${CASE_TAG}_job3019273_${SLURM_ARRAY_TASK_ID}"
    ;;
  2)
    CASE_TAG="t24x1"
    CFG_OPTS="train_pipeline.0.clip_len=24 train_pipeline.0.frame_interval=1 \
      val_pipeline.0.clip_len=24   val_pipeline.0.frame_interval=1 \
      test_pipeline.0.clip_len=24  test_pipeline.0.frame_interval=1"
    RUN_DIR="$work_dir/${CASE_TAG}_job3019374_${SLURM_ARRAY_TASK_ID}"
    ;;
  3)
    CASE_TAG="t32x1"
    CFG_OPTS="train_pipeline.0.clip_len=32 train_pipeline.0.frame_interval=1 \
      val_pipeline.0.clip_len=32   val_pipeline.0.frame_interval=1 \
      test_pipeline.0.clip_len=32  test_pipeline.0.frame_interval=1"
    RUN_DIR="$work_dir/${CASE_TAG}_job3007432_${SLURM_ARRAY_TASK_ID}"
    ;;
  *)
    echo "Geçersiz array index: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac


mkdir -p "$RUN_DIR"


echo "==> CONFIG     : $config"
echo "==> EXPERIMENT : $CASE_TAG"
echo "==> WORK DIR   : $RUN_DIR"

# variables
res="_256x256"
dataset_root="/arf/scratch/zgokce/data"
exp_name="small${res}"
work_dir="/arf/scratch/zgokce/workdir/swin/WLASL100/${exp_name}"

dump="${work_dir}/result.pkl"
config="./configs/SLR/swin_wlasl100_256x256/finetune_WLASL100_swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py"

#test


srun python ./tools/test.py "$config" "$ckpt_path" \
  --cfg-options work_dir="$RUN_DIR" dataset_root="$dataset_root" $CFG_OPTS \
  --dump "$dump"

exit
