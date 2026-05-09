#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A zgokce
#SBATCH -J model_complexity
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00-02:00
#SBATCH --output=/arf/scratch/zgokce/logs/model_complexity/slurm-%x-job%j-%t.out
#SBATCH --error=/arf/scratch/zgokce/logs/model_complexity/slurm-%x-job%j-%t.err

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

mkdir -p /arf/scratch/zgokce/logs/model_complexity

OUT="/arf/scratch/zgokce/workdir/model_complexity/model_complexity.csv"
mkdir -p "$(dirname $OUT)"

srun python tools/analysis_tools/measure_model_complexity.py \
  --device cuda:0 \
  --warmup 50 \
  --iters 100 \
  --out "$OUT"

exit
