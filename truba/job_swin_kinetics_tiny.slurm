#!/bin/bash
#SBATCH -p barbun-cuda                 # GPU içeren uygun kuyruk
#SBATCH -A zgokce                   # Kullanıcı adı
#SBATCH -J swin_1gpu        # İşin adı
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

wdir=/arf/home/zgokce/code/mmaction2_swin  #Uygulama için çalıştırılacak dosyaların adresleri.
cd $wdirs

conda activate open-mmlab
srun python ./tools/test.py "./configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py"  "ckpt/swin_tiny_patch244_window877_kinetics400_1k.pth"


exit
