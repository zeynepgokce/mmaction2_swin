#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A zgokce
#SBATCH -J uniformer_base_wlasl
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
srun python ./tools/train.py "configs/recognition/uniformerv2/finetune_WLASL100_uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb.py"  --cfg-options work_dir="/arf/scratch/zgokce/workdir/WLASL100/uniformerv2/base/" dataset_root="/arf/home/zgokce/code/videomae_v2/data" train_dataloader.batch_size=4 val_dataloader.batch_size=4

srun python ./tools/test.py "configs/recognition/uniformerv2/finetune_WLASL100_uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb_test.py"  "/arf/scratch/zgokce/workdir/WLASL100/uniformerv2/base/epoch_45.pth" --cfg-options work_dir="/arf/scratch/zgokce/workdir/WLASL100/uniformerv2/base/" dataset_root="/arf/home/zgokce/code/videomae_v2/data"  --dump /arf/scratch/zgokce/workdir/WLASL100/uniformerv2/base/result.pkl

exit
