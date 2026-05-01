#!/usr/bin/env python3
"""bench_report.py — parse mmaction2 test output and append to benchmark report.

Usage (called from SLURM scripts after tools/test.py):
    python tools/bench_report.py \
        --model swin \
        --dataset wlasl100 \
        --train_res 256 \
        --test_type 256 \
        --config configs/SLR/bench_swin/... \
        --ckpt /path/to/best.pth \
        --eval_log /path/to/eval.log \
        --report_dir /arf/scratch/zgokce/bench/reports \
        [--train_log /path/to/train.log]

Outputs:
    {report_dir}/report_{model}_{dataset}.txt   — detailed per-run block
    {report_dir}/report_GLOBAL_SUMMARY.tsv      — one TSV row per eval run
"""

import argparse
import os
import re
import socket
import subprocess
import sys
from datetime import datetime


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True,
                   help='Model name, e.g. swin or uniformer')
    p.add_argument('--dataset', required=True,
                   help='Dataset name, e.g. wlasl100 or aslcitizen100')
    p.add_argument('--train_res', required=True,
                   help='Training resolution variant, e.g. 256 or 64_resize256')
    p.add_argument('--test_type', required=True,
                   help='Test input type: 256, 64, or SR')
    p.add_argument('--config', required=True,
                   help='Config file path used for evaluation')
    p.add_argument('--ckpt', required=True,
                   help='Checkpoint path used for evaluation')
    p.add_argument('--eval_log', required=True,
                   help='Path to eval stdout/stderr log file')
    p.add_argument('--report_dir', required=True,
                   help='Directory where report files are written')
    p.add_argument('--train_log', default=None,
                   help='Path to training log (optional, for hyperparameter extraction)')
    return p.parse_args()


# ── Metric parsing ─────────────────────────────────────────────────────────

def parse_metrics(log_path):
    """Extract top1 / top5 accuracy from mmaction2 test output."""
    top1 = top5 = None
    if not os.path.isfile(log_path):
        return top1, top5
    with open(log_path, 'r', errors='replace') as f:
        text = f.read()
    # mmaction2 prints something like:
    #   acc/top1: 0.6543   acc/top5: 0.9012
    #   top1_acc: 0.6543   top5_acc: 0.9012   (older format)
    m1 = re.search(r'acc/top1\s*[:\s]+([0-9.]+)', text)
    m5 = re.search(r'acc/top5\s*[:\s]+([0-9.]+)', text)
    if not m1:
        m1 = re.search(r'top1_acc\s*[:\s]+([0-9.]+)', text)
    if not m5:
        m5 = re.search(r'top5_acc\s*[:\s]+([0-9.]+)', text)
    if m1:
        top1 = float(m1.group(1))
    if m5:
        top5 = float(m5.group(1))
    return top1, top5


def parse_train_hyperparams(train_log_path, config_path):
    """Best-effort extraction of key hyperparameters from config or train log."""
    lr = wd = scheduler = epochs = batch = 'N/A'
    if config_path and os.path.isfile(config_path):
        with open(config_path, 'r', errors='replace') as f:
            text = f.read()
        m = re.search(r'\blr\s*=\s*([0-9e.+-]+)', text)
        if m:
            lr = m.group(1)
        m = re.search(r'weight_decay\s*=\s*([0-9e.+-]+)', text)
        if m:
            wd = m.group(1)
        m = re.search(r'max_epochs\s*=\s*(\d+)', text)
        if m:
            epochs = m.group(1)
        m = re.search(r'batch_size\s*=\s*(\d+)', text)
        if m:
            batch = m.group(1)
        if 'CosineAnnealingLR' in text:
            scheduler = 'CosineAnnealingLR'
        elif 'StepLR' in text:
            scheduler = 'StepLR'
    return dict(lr=lr, wd=wd, scheduler=scheduler, epochs=epochs, batch=batch)


def get_git_hash(repo_dir):
    try:
        result = subprocess.run(
            ['git', '-C', repo_dir, 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


# ── Report writing ─────────────────────────────────────────────────────────

TSV_HEADER = (
    'timestamp\thostname\tgit_hash\tmodel\tdataset\ttrain_res\ttest_type\t'
    'top1\ttop5\tckpt\tconfig\tlr\twd\tscheduler\tepochs\tbatch\t'
    'frames\tinput_to_model\tsource_res\n'
)


def append_detailed_report(report_path, info):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    block = (
        '\n' + '=' * 80 + '\n'
        f"TIMESTAMP   : {info['timestamp']}\n"
        f"HOSTNAME    : {info['hostname']}\n"
        f"GIT HASH    : {info['git_hash']}\n"
        f"MODEL       : {info['model']}\n"
        f"DATASET     : {info['dataset']}\n"
        f"TRAIN RES   : {info['train_res']}\n"
        f"TEST TYPE   : {info['test_type']}\n"
        f"TOP1 ACC    : {info['top1']}\n"
        f"TOP5 ACC    : {info['top5']}\n"
        f"CHECKPOINT  : {info['ckpt']}\n"
        f"CONFIG      : {info['config']}\n"
        f"FRAMES      : {info['frames']}\n"
        f"INPUT→MODEL : {info['input_to_model']}\n"
        f"SOURCE RES  : {info['source_res']}\n"
        f"LR          : {info['lr']}\n"
        f"WD          : {info['wd']}\n"
        f"SCHEDULER   : {info['scheduler']}\n"
        f"EPOCHS      : {info['epochs']}\n"
        f"BATCH SIZE  : {info['batch']}\n"
        '=' * 80 + '\n'
    )
    with open(report_path, 'a') as f:
        f.write(block)
    print(f'[bench_report] Appended detailed report → {report_path}')


def append_tsv_row(tsv_path, info):
    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
    # Write header if file is new / empty
    write_header = not os.path.isfile(tsv_path) or os.path.getsize(tsv_path) == 0
    row = (
        f"{info['timestamp']}\t{info['hostname']}\t{info['git_hash']}\t"
        f"{info['model']}\t{info['dataset']}\t{info['train_res']}\t"
        f"{info['test_type']}\t{info['top1']}\t{info['top5']}\t"
        f"{info['ckpt']}\t{info['config']}\t"
        f"{info['lr']}\t{info['wd']}\t{info['scheduler']}\t"
        f"{info['epochs']}\t{info['batch']}\t"
        f"{info['frames']}\t{info['input_to_model']}\t{info['source_res']}\n"
    )
    with open(tsv_path, 'a') as f:
        if write_header:
            f.write(TSV_HEADER)
        f.write(row)
    print(f'[bench_report] Appended TSV row → {tsv_path}')


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    top1, top5 = parse_metrics(args.eval_log)
    hyperparams = parse_train_hyperparams(args.train_log, args.config)
    git_hash = get_git_hash(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    # Derived fields
    source_res = '64x64' if args.test_type in ('64',) else (
        'SR_256x256' if args.test_type == 'SR' else '256x256')
    input_to_model = '224x224'  # always crop to 224 before model

    info = dict(
        timestamp=datetime.now().isoformat(timespec='seconds'),
        hostname=socket.gethostname(),
        git_hash=git_hash,
        model=args.model,
        dataset=args.dataset,
        train_res=args.train_res,
        test_type=args.test_type,
        top1=top1 if top1 is not None else 'PARSE_FAILED',
        top5=top5 if top5 is not None else 'N/A',
        ckpt=args.ckpt,
        config=args.config,
        frames=16,
        input_to_model=input_to_model,
        source_res=source_res,
        **hyperparams,
    )

    # Print summary to stdout
    print(f'\n{"─"*60}')
    print(f'[bench_report] {args.model} | {args.dataset} | '
          f'train={args.train_res} | test={args.test_type}')
    print(f'  top1={info["top1"]}  top5={info["top5"]}')
    print(f'  ckpt={args.ckpt}')
    print(f'{"─"*60}\n')

    os.makedirs(args.report_dir, exist_ok=True)

    detailed_path = os.path.join(
        args.report_dir, f'report_{args.model}_{args.dataset}.txt')
    tsv_path = os.path.join(args.report_dir, 'report_GLOBAL_SUMMARY.tsv')

    append_detailed_report(detailed_path, info)
    append_tsv_row(tsv_path, info)

    if top1 is None:
        print('[bench_report] WARNING: Could not parse top1 accuracy from log. '
              'Check eval_log for mmaction2 output format.', file=sys.stderr)


if __name__ == '__main__':
    main()
