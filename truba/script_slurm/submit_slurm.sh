#!/bin/bash
# Kullanım örneği:
# bash /arf/home/zgokce/code/mmaction2_swin/truba/script_slurm/wlasl100_256x256/submit_slurm.sh /arf/home/zgokce/code/mmaction2_swin/truba/script_slurm/wlasl100_256x256/...
# SLURM betikleri sırayla gönderme scripti
# Kullanım: ./submit_slurm.sh /path/to/directory

# Kontrol: argüman verildi mi?
if [ $# -eq 0 ]; then
    echo "Kullanım: $0 /path/to/directory"
    echo "Örnek: $0 /arf/home/zgokce/jobs"
    exit 1
fi

SCRIPT_DIR="$1"

# Kontrol: dizin var mı?
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "Hata: $SCRIPT_DIR dizini bulunamadı!"
    exit 1
fi

# Eşleşen dosyaları bul
shopt -s nullglob
SLURM_FILES=("$SCRIPT_DIR"/job*.slurm)

# Kontrol: eşleşen dosya var mı?
if [ ${#SLURM_FILES[@]} -eq 0 ]; then
    echo "Hata: $SCRIPT_DIR dizininde job_*.slurm dosyası bulunamadı!"
    exit 1
fi

echo "Bulundu: ${#SLURM_FILES[@]} adet SLURM dosyası"
echo "=========================================="

# Dosyaları sıralı bir şekilde gönder
for slurm_file in "${SLURM_FILES[@]}"; do
    if [ -f "$slurm_file" ]; then
        filename=$(basename "$slurm_file")
        echo "Gönderiliyor: $filename"

        # sbatch komutunu çalıştır
        job_id=$(sbatch "$slurm_file" | awk '{print $NF}')

        if [ $? -eq 0 ]; then
            echo "✓ Başarılı - Job ID: $job_id"
        else
            echo "✗ Hata: $filename gönderilemedi!"
        fi

        echo "----------------------------------------"

        # Betikleri çok hızlı peşpeşe göndermemek için kısa bir bekleme
        sleep 2
    fi
done

echo "=========================================="
echo "✓ Tüm SLURM betikleri gönderildi!"
echo ""
echo "İş durumunu kontrol etmek için:"
echo "  squeue -u $(whoami)"
echo ""
echo "Tüm işleri görmek için:"
echo "  squeue"