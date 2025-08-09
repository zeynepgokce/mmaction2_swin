#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A zgokce
#SBATCH -J swin_base_wlasl_2gpu
#SBATCH --gres=gpu:2               # Tek node'da 2 GPU
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --time=03-00:00
#SBATCH --output=logs/slurm-%x-%j-%t.out
#SBATCH --error=logs/slurm-%x-%j-%t.err

#export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)


# CUDA bellek yönetimi için fragmentasyonu azalt
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# Dağıtık eğitim için adres ve port
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Resolved master address:"
scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + RANDOM % 20000))
export NCCL_DEBUG=INFO

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
srun python ./tools/train.py "./configs/recognition/swin/finetune_WLASL100_swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py"  --cfg-options work_dir="/arf/scratch/zgokce/workdir/swin/WLASL100/base/" dataset_root="/arf/scratch/zgokce/data" load_from="ckpt/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-182ec6cc.pth"  train_dataloader.batch_size=8 val_dataloader.batch_size=8 train_dataloader.num_workers=16 val_dataloader.num_workers=8

srun python ./tools/test.py "./configs/recognition/swin/finetune_WLASL100_swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py" "/arf/scratch/zgokce/workdir/swin/WLASL100/base/epoch_30.pth" --cfg-options work_dir="/arf/scratch/zgokce/workdir/swin/WLASL100/base/" dataset_root="/arf/scratch/zgokce/data"  test_dataloader.num_workers=4 --dump /arf/scratch/zgokce/workdir/swin/WLASL100/base/result.pkl

exit
