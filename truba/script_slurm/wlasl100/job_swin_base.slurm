#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A zgokce
#SBATCH -J swin_base_wlasl
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
srun python ./tools/train.py "./configs/recognition/swin/finetune_WLASL100_swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py"  --cfg-options work_dir="/arf/scratch/zgokce/workdir/WLASL100/swin/base/" dataset_root="/arf/home/zgokce/code/videomae_v2/data" load_from="ckpt/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-182ec6cc.pth" train_dataloader.batch_size=4 val_dataloader.batch_size=4

srun python ./tools/test.py "./configs/recognition/swin/finetune_ASL_swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py" "/arf/scratch/zgokce/workdir/WLASL100/swin/base/epoch_45.pth" --cfg-options work_dir="/arf/scratch/zgokce/workdir/wlasl/swin/base/" dataset_root="/arf/home/zgokce/code/videomae_v2/data"  --dump /arf/scratch/zgokce/workdir/WLASL100/swin/base/result.pkl

exit
