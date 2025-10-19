#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A zgokce
#SBATCH -J uniformer_base_wlasl_hr
#SBATCH --gres=gpu:1                # Tek node'da 1 GPU
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=03-00:00
#SBATCH --output=logs/slurm-%x-%j-%t.out
#SBATCH --error=logs/slurm-%x-%j-%t.err

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


# variables
hr="_hr"
dataset_root="/arf/scratch/zgokce/data"
exp_name="base${hr}"
work_dir="/arf/scratch/zgokce/workdir/uniformerv2/WLASL100/${exp_name}"
ckpt_path="${work_dir}/epoch_30.pth"
dump="${work_dir}/result.pkl"
config="./configs/recognition/uniformerv2/finetune_WLASL100_uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb${hr}.py"


srun python ./tools/train.py "$config" \
  --cfg-options work_dir="$work_dir" dataset_root="$dataset_root"

srun python ./tools/test.py "$config" "$ckpt_path" \
  --cfg-options work_dir="$work_dir" dataset_root="$dataset_root" \
  --dump "$dump"
exit



