#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A zgokce
#SBATCH -J swin_large_multigpu
#SBATCH --gres=gpu:1              # Tek node'da 1 GPU
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=03-00:00
#SBATCH --output=logs/slurm-%x-%j-%t.out
#SBATCH --error=logs/slurm-%x-%j-%t.err

set -x  # çalıştırılan komutları ekranda göster

source /etc/profile.d/modules.sh
module purge
module load lib/cuda/11.8
module load miniconda3

#export MASTER_PORT=$((12000 + $RANDOM % 20000))
#export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

export PYTHONPATH=/arf/home/zgokce/miniconda3/envs/open-mmlab/lib/python3.7/site-packages

echo "NODE: $(hostname)"
nvidia-smi

# environment
wdir=/arf/home/zgokce/code/mmaction2_swin
cd $wdir
conda activate open-mmlab

#train test scripts run

srun python ./tools/train.py "./configs/recognition/swin/finetune_ASL_swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py"  --cfg-options work_dir="/arf/scratch/zgokce/workdir/swin/large/" dataset_root="/arf/home/zgokce/code/videomae_v2/data" load_from="/arf/scratch/zgokce/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-78ad8b11.pth" train_dataloader.batch_size=2 val_dataloader.batch_size=2

srun python ./tools/test.py "./configs/recognition/swin/finetune_ASL_swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py" "/arf/scratch/zgokce/workdir/swin/large/epoch_45.pth" --cfg-options work_dir="/arf/scratch/zgokce/workdir/swin/large/" dataset_root="/arf/home/zgokce/code/videomae_v2/data"  --dump /arf/scratch/zgokce/workdir/swin/large/result.pkl

exit
