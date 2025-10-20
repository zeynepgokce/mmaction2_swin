#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# Orijinal şablon dosyasını oku
template_file = "/home/zeynep/Thesis/code/mmaction2/truba/script_slurm/wlasl100_64x64/swin_base/job_swin_base.slurm"
slurm_folder = "/home/zeynep/Thesis/code/mmaction2/truba/script_slurm/wlasl100_64x64/swin_base"

#swin small
experiments = [
    {
        "deney_ismi": "exp_lr1e-03_c32x2_bs2",
        "cfg_opts": "optim_wrapper.optimizer.lr=0.001 train_pipeline.0.clip_len=32 train_pipeline.0.frame_interval=2 val_pipeline.0.clip_len=32 val_pipeline.0.frame_interval=2 test_pipeline.0.clip_len=32 test_pipeline.0.frame_interval=2 train_dataloader.batch_size=2 val_dataloader.batch_size=2"
    },
    {
        "deney_ismi": "exp_lr5e-04_c32x2_bs2",
        "cfg_opts": "optim_wrapper.optimizer.lr=0.0005 train_pipeline.0.clip_len=32 train_pipeline.0.frame_interval=2 val_pipeline.0.clip_len=32 val_pipeline.0.frame_interval=2 test_pipeline.0.clip_len=32 test_pipeline.0.frame_interval=2 train_dataloader.batch_size=2 val_dataloader.batch_size=2"
    },
    {
        "deney_ismi": "exp_lr1e-03_c32x2_bs4",
        "cfg_opts": "optim_wrapper.optimizer.lr=0.001 train_pipeline.0.clip_len=32 train_pipeline.0.frame_interval=2 val_pipeline.0.clip_len=32 val_pipeline.0.frame_interval=2 test_pipeline.0.clip_len=32 test_pipeline.0.frame_interval=2 train_dataloader.batch_size=4 val_dataloader.batch_size=4"
    },
    {
        "deney_ismi": "exp_lr5e-04_c32x2_bs4",
        "cfg_opts": "optim_wrapper.optimizer.lr=0.0005 train_pipeline.0.clip_len=32 train_pipeline.0.frame_interval=2 val_pipeline.0.clip_len=32 val_pipeline.0.frame_interval=2 test_pipeline.0.clip_len=32 test_pipeline.0.frame_interval=2 train_dataloader.batch_size=4 val_dataloader.batch_size=4"
    },
    {
        "deney_ismi": "exp_lr1e-03_c16x2_bs2",
        "cfg_opts": "optim_wrapper.optimizer.lr=0.001 train_pipeline.0.clip_len=16 train_pipeline.0.frame_interval=2 val_pipeline.0.clip_len=16 val_pipeline.0.frame_interval=2 test_pipeline.0.clip_len=16 test_pipeline.0.frame_interval=2 train_dataloader.batch_size=2 val_dataloader.batch_size=2"
    },
    {
        "deney_ismi": "exp_lr5e-04_c16x2_bs2",
        "cfg_opts": "optim_wrapper.optimizer.lr=0.0005 train_pipeline.0.clip_len=16 train_pipeline.0.frame_interval=2 val_pipeline.0.clip_len=16 val_pipeline.0.frame_interval=2 test_pipeline.0.clip_len=16 test_pipeline.0.frame_interval=2 train_dataloader.batch_size=2 val_dataloader.batch_size=2"
    },
    {
        "deney_ismi": "exp_lr1e-03_c16x2_bs4",
        "cfg_opts": "optim_wrapper.optimizer.lr=0.001 train_pipeline.0.clip_len=16 train_pipeline.0.frame_interval=2 val_pipeline.0.clip_len=16 val_pipeline.0.frame_interval=2 test_pipeline.0.clip_len=16 test_pipeline.0.frame_interval=2 train_dataloader.batch_size=4 val_dataloader.batch_size=4"
    },
    {
        "deney_ismi": "exp_lr5e-04_c16x2_bs4",
        "cfg_opts": "optim_wrapper.optimizer.lr=0.0005 train_pipeline.0.clip_len=16 train_pipeline.0.frame_interval=2 val_pipeline.0.clip_len=16 val_pipeline.0.frame_interval=2 test_pipeline.0.clip_len=16 test_pipeline.0.frame_interval=2 train_dataloader.batch_size=4 val_dataloader.batch_size=4"
    },
    {
        "deney_ismi": "exp_lr1e-03_c16x1_bs2",
        "cfg_opts": "optim_wrapper.optimizer.lr=0.001 train_pipeline.0.clip_len=16 train_pipeline.0.frame_interval=1 val_pipeline.0.clip_len=16 val_pipeline.0.frame_interval=1 test_pipeline.0.clip_len=16 test_pipeline.0.frame_interval=1 train_dataloader.batch_size=2 val_dataloader.batch_size=2"
    },
    {
        "deney_ismi": "exp_lr5e-04_c16x1_bs2",
        "cfg_opts": "optim_wrapper.optimizer.lr=0.0005 train_pipeline.0.clip_len=16 train_pipeline.0.frame_interval=1 val_pipeline.0.clip_len=16 val_pipeline.0.frame_interval=1 test_pipeline.0.clip_len=16 test_pipeline.0.frame_interval=1 train_dataloader.batch_size=2 val_dataloader.batch_size=2"
    },
    {
        "deney_ismi": "exp_lr1e-03_c16x1_bs4",
        "cfg_opts": "optim_wrapper.optimizer.lr=0.001 train_pipeline.0.clip_len=16 train_pipeline.0.frame_interval=1 val_pipeline.0.clip_len=16 val_pipeline.0.frame_interval=1 test_pipeline.0.clip_len=16 test_pipeline.0.frame_interval=1 train_dataloader.batch_size=4 val_dataloader.batch_size=4"
    },
    {
        "deney_ismi": "exp_lr5e-04_c16x1_bs4",
        "cfg_opts": "optim_wrapper.optimizer.lr=0.0005 train_pipeline.0.clip_len=16 train_pipeline.0.frame_interval=1 val_pipeline.0.clip_len=16 val_pipeline.0.frame_interval=1 test_pipeline.0.clip_len=16 test_pipeline.0.frame_interval=1 train_dataloader.batch_size=4 val_dataloader.batch_size=4"
    },
]



with open(template_file, 'r', encoding='utf-8') as f:
	template_content = f.read()

# Her deney için dosya oluştur
for exp in experiments:
	deney_ismi = exp["deney_ismi"]
	cfg_opts = exp["cfg_opts"]

	# Çıkış dosya adı
	output_filename = f"{slurm_folder}/job_swin_base_wlasl100_64x64_{deney_ismi}.slurm"

	# Şablonu kopyala
	modified_content = template_content

	# 1. #SBATCH -J satırını güncelle
	old_j_line = "#SBATCH -J swin_base_wlasl100_64x64"
	new_j_line = f"#SBATCH -J swin_base_wlasl100_64x64_{deney_ismi}"
	modified_content = modified_content.replace(old_j_line, new_j_line)

	# 2. EXP_NAME değişkenini güncelle
	old_exp_name = 'EXP_NAME="base"'
	new_exp_name = f'EXP_NAME="{deney_ismi}"'
	modified_content = modified_content.replace(old_exp_name, new_exp_name)

	# 3. CFG_OPTS değişkenini güncelle
	old_cfg_opts = 'CFG_OPTS="train_pipeline.0.clip_len=32 train_pipeline.0.frame_interval=2  val_pipeline.0.clip_len=32   val_pipeline.0.frame_interval=2 test_pipeline.0.clip_len=32  test_pipeline.0.frame_interval=1"'
	new_cfg_opts = f'CFG_OPTS="{cfg_opts}"'
	modified_content = modified_content.replace(old_cfg_opts, new_cfg_opts)

	# Dosyayı yaz
	with open(output_filename, 'w', encoding='utf-8') as f:
		f.write(modified_content)

	# Dosyayı çalıştırılabilir yap
	os.chmod(output_filename, 0o755)

	print(f"✓ {output_filename} oluşturuldu")

print(f"\n✓ Toplam {len(experiments)} adet .slurm dosyası başarıyla oluşturuldu!")