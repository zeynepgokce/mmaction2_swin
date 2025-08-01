#!/bin/bash
#SBATCH -p barbun-cuda                 # GPU içeren uygun kuyruk
#SBATCH -A zgokce                   # Kullanıcı adı
#SBATCH -J swin_tiny_wlasl       # İşin adı
#SBATCH --gres=gpu:1                # Tek node'da 1 GPU
#SBATCH --nodes 1                        # Tek node
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=03-00:00             # Maksimum süre: 3 gün
#SBATCH --output=logs/slurm-%x-%j-%t.out  # %x=job name, %j=job ID, %t=task ID
#SBATCH --error=logs/slurm-%x-%j-%t.err

source /etc/profile.d/modules.sh
module purge
module load lib/cuda/11.8
module load miniconda3

echo "NODE: $(hostname)"
nvidia-smi

wdir=/arf/home/zgokce/code/mmaction2_swin #Uygulama için çalıştırılacak dosyaların adresleri.
cd $wdir

export PYTHONPATH=/arf/home/zgokce/miniconda3/envs/open-mmlab/lib/python3.7/site-packages

conda activate open-mmlab

srun python ./tools/train.py "./configs/recognition/swin/finetune_WLASL100_swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py"  --cfg-options work_dir="/arf/scratch/zgokce/workdir/swin/WLASL100/tiny/" dataset_root="/arf/scratch/zgokce/data" train_dataloader.batch_size=4 val_dataloader.batch_size=4 train_dataloader.num_workers=16 val_dataloader.num_workers=8

srun python ./tools/test.py "./configs/recognition/swin/finetune_WLASL100_swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py" "/arf/scratch/zgokce/workdir/swin/WLASL100/tiny/epoch_45.pth" --cfg-options work_dir="/arf/scratch/zgokce/workdir/swin/WLASL100/tiny/" dataset_root="/arf/scratch/zgokce/data"  --dump /arf/scratch/zgokce/workdir/swin/WLASL100/tiny/result.pkl

exit
