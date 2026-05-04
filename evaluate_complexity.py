#!/usr/bin/env python3
"""
Model complexity and speed evaluation for mmaction2 video models.

Run with:
    python evaluate_complexity.py <config>
    python evaluate_complexity.py <config> --checkpoint /path/to/best.pth
    python evaluate_complexity.py <config> --checkpoint /path/to/best.pth --device cpu

Metrics reported
────────────────
TRAIN MODE  : total/trainable params (manual + fvcore cross-check), model size, ckpt size
EVAL MODE   : same parameter metrics
EVAL MODE   : inference time (ms) and GFLOPs  ← no gradients, CUDA synced
"""

import argparse
import os
import sys
import time
import glob
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

# mmaction2 / mmengine
from mmengine.config import Config
from mmaction.apis import init_recognizer

# fvcore
from fvcore.nn import FlopCountAnalysis, parameter_count

PROJECT_ROOT  = "/home/zeynep/Thesis/code/mmaction2"
WARMUP_RUNS   = 20
TIMED_RUNS    = 100

# ─────────────────────────────────────────────────────────────────────────────
#  HARDCODED DEFAULTS  ── keyed by config filename stem
#  CLI args (--checkpoint, --data-root, --ann-file) override these.
# ─────────────────────────────────────────────────────────────────────────────

_D = "/media/zeynep/SSD/phd/datasets/ASL_Citizen/subsets/"
_W = "/home/zeynep/Thesis/code/mmaction2/workdir"

DEFAULTS = {
    # ── Swin-Base / ASLCitizen100 ──────────────────────────────────────────
    "swin_aslcitizen100_train256": dict(
        checkpoint = f"{_W}/swin_aslcitizen100_train256",
        data_root  = f"{_D}/ASLCitizen100_videos_256x256/test",
        ann_file   = f"{_D}/ASLCitizen100_videos_256x256/test_aslcitizen100_mm2.txt",
    ),
    "swin_aslcitizen100_train64_resize256": dict(
        checkpoint = f"{_W}/swin_aslcitizen100_train64_resize256",
        data_root  = f"{_D}/ASLCitizen100_videos_256x256_bilinear/test",
        ann_file   = f"{_D}/ASLCitizen100_videos_256x256_bilinear/test_aslcitizen100_mm2.txt",
    ),
    # ── Swin-Base / WLASL100 ──────────────────────────────────────────────
    "swin_wlasl100_train256": dict(
        checkpoint = f"{_W}/swin_wlasl100_train256",
        data_root  = f"{_D}/WLASL100_videos_256x256/test",
        ann_file   = f"{_D}/WLASL100_videos_256x256/test_wlasl100_mm2.txt",
    ),
    "swin_wlasl100_train64_resize256": dict(
        checkpoint = f"{_W}/swin_wlasl100_train64_resize256",
        data_root  = f"{_D}/wlasl100_videos_64x64/test",
        ann_file   = f"{_D}/wlasl100_videos_64x64/test_wlasl100_mm2.txt",
    ),
    # ── UniformerV2 / ASLCitizen100 ──────────────────────────────────────
    "uniformer_aslcitizen100_train256": dict(
        checkpoint = f"{_W}/uniformer_aslcitizen100_train256",
        data_root  = f"{_D}/ASLCitizen100_videos_256x256/test",
        ann_file   = f"{_D}/ASLCitizen100_videos_256x256/test_aslcitizen100_mm2.txt",
    ),
    "uniformer_aslcitizen100_train64_resize256": dict(
        checkpoint = f"{_W}/uniformer_aslcitizen100_train64_resize256",
        data_root  = f"{_D}/ASLCitizen100_videos_256x256_bilinear/test",
        ann_file   = f"{_D}/ASLCitizen100_videos_256x256_bilinear/test_aslcitizen100_mm2.txt",
    ),
    # ── UniformerV2 / WLASL100 ───────────────────────────────────────────
    "uniformer_wlasl100_train256": dict(
        checkpoint = f"{_W}/uniformer_wlasl100_train256",
        data_root  = f"{_D}/WLASL100_videos_256x256/test",
        ann_file   = f"{_D}/WLASL100_videos_256x256/test_wlasl100_mm2.txt",
    ),
    "uniformer_wlasl100_train64_resize256": dict(
        checkpoint = f"{_W}/uniformer_wlasl100_train64_resize256",
        data_root  = f"{_D}/wlasl100_videos_64x64/test",
        ann_file   = f"{_D}/wlasl100_videos_64x64/test_wlasl100_mm2.txt",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT AUTO-SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def find_best_checkpoint(config_path: str, root: str) -> str:
    """
    Search order:
      1) DEFAULTS[stem]['checkpoint'] dir (TRUBA workdir)
      2) work_dirs/<config_stem>/         (local workdir)
      3) work_dirs/**/                    (broader local search)
      4) epoch_*.pth fallback
    """
    stem = os.path.splitext(os.path.basename(config_path))[0]

    search_dirs = []
    if stem in DEFAULTS and DEFAULTS[stem].get("checkpoint"):
        search_dirs.append(DEFAULTS[stem]["checkpoint"])
    search_dirs += [
        os.path.join(root, "work_dirs", stem),
        os.path.join(root, "work_dirs"),
        os.path.join(root, "truba", "workdir"),
    ]

    # Priority 1 & 2 – any best_*.pth
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        hits = glob.glob(
            os.path.join(search_dir, "**", "best_*.pth"), recursive=True
        )
        if hits:
            chosen = max(hits, key=os.path.getmtime)
            print(f"[ckpt] Auto-selected best checkpoint:\n       {chosen}")
            return chosen

    # Priority 3 – latest epoch checkpoint
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        hits = glob.glob(
            os.path.join(search_dir, "**", "epoch_*.pth"), recursive=True
        )
        if hits:
            def _epoch_num(p):
                try:
                    return int(
                        os.path.basename(p).replace("epoch_", "").replace(".pth", "")
                    )
                except ValueError:
                    return 0

            chosen = max(hits, key=_epoch_num)
            print(f"[ckpt] Auto-selected epoch checkpoint:\n       {chosen}")
            return chosen

    raise FileNotFoundError(
        f"No .pth checkpoint found.\n"
        f"Searched directories: {search_dirs}\n"
        f"Please place a checkpoint under work_dirs/{stem}/"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT LOADING  (handles common key wrappers)
# ─────────────────────────────────────────────────────────────────────────────

def load_weights(model: nn.Module, ckpt_path: str, device) -> nn.Module:
    raw = torch.load(ckpt_path, map_location=device)
    state = raw
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in raw:
                state = raw[key]
                print(f"[ckpt] Unwrapped checkpoint key: '{key}'")
                break

    # Strip DataParallel / DistributedDataParallel prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(
            f"[ckpt] {len(missing)} missing key(s) "
            f"(first 3): {missing[:3]}"
        )
    if unexpected:
        print(
            f"[ckpt] {len(unexpected)} unexpected key(s) "
            f"(first 3): {unexpected[:3]}"
        )
    print("[ckpt] Weights loaded successfully.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING  (real test sample → fallback to synthetic)
# ─────────────────────────────────────────────────────────────────────────────

def get_input_tensor(cfg, device) -> torch.Tensor:
    """
    Returns float32 tensor (1, num_clips, C, T, H, W) on `device`.

    Pipeline (from config):
        DecordInit → UniformSample(clip_len=16) → DecordDecode
        → Resize(64×64) → Resize(-1,256) → CenterCrop(224)
        → FormatShape(NCTHW) → PackActionInputs
    Expected shape: (1, 1, 3, 16, 224, 224) with num_clips=1

    Falls back to a synthetic tensor if videos are not accessible.
    """
    try:
        from mmengine.registry import DATASETS

        ds_cfg = cfg.test_dataloader.dataset
        dataset = DATASETS.build(ds_cfg)

        sample = dataset[0]          # dict: {'inputs': Tensor, 'data_samples': ...}
        inp = sample["inputs"]       # (num_clips, C, T, H, W)  e.g. (1,3,16,224,224)

        if isinstance(inp, list):
            inp = torch.stack(inp)   # handle list-of-tensors

        # Ensure 5-D (num_clips, C, T, H, W) → add batch dim → (1, num_clips, C, T, H, W)
        if inp.dim() == 4:
            inp = inp.unsqueeze(0).unsqueeze(0)
        elif inp.dim() == 5:
            inp = inp.unsqueeze(0)

        inp = inp.float().to(device)
        print(f"[data] Real sample loaded from test set.  shape={tuple(inp.shape)}")
        return inp

    except Exception as exc:
        fallback = (1, 1, 3, 16, 224, 224)
        print(f"[data] Real data unavailable "
              f"({type(exc).__name__}: {exc})")
        print(f"[data] Falling back to synthetic tensor {fallback}")
        # Scale to uint8 range so normalization makes sense
        return torch.randint(0, 256, fallback, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
#  BACKBONE+HEAD WRAPPER  (for FLOPs & timing – skips data_preprocessor)
# ─────────────────────────────────────────────────────────────────────────────

class BackboneHeadWrapper(nn.Module):
    """
    Wraps SwinTransformer3D backbone + I3DHead.
    Accepts a pre-normalized (B, C, T, H, W) tensor directly,
    bypassing ActionDataPreprocessor (normalization only – negligible FLOPs).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.cls_head  = model.cls_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        feat = self.backbone(x)   # → (B, C', T', H', W')
        return self.cls_head(feat)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def param_stats(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def normalize_input(x: torch.Tensor, device) -> torch.Tensor:
    """Apply ActionDataPreprocessor normalization (mean/std from config)."""
    mean = torch.tensor(
        [123.675, 116.28, 103.53], device=device
    ).view(1, 3, 1, 1, 1)
    std = torch.tensor(
        [58.395, 57.12, 57.375], device=device
    ).view(1, 3, 1, 1, 1)
    return (x.clamp(0.0, 255.0) - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Model complexity & speed evaluation for mmaction2 video models"
    )
    p.add_argument("config", help="Config file path")
    p.add_argument(
        "--checkpoint", "-c", default=None,
        help="Path to .pth checkpoint. If omitted, auto-searches work_dirs/")
    p.add_argument(
        "--data-root", default=None,
        help="Override test video directory (data_prefix.video in config)")
    p.add_argument(
        "--ann-file", default=None,
        help="Override test annotation file path (ann_file in config)")
    p.add_argument(
        "--device", default=None,
        help="cuda:0 | cpu  (default: cuda if available, else cpu)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # ── Config ───────────────────────────────────────────────────────────────
    config_path = args.config
    assert os.path.isfile(config_path), f"Config not found:\n  {config_path}"
    cfg = Config.fromfile(config_path)

    # Apply DEFAULTS for this config (CLI args take priority)
    stem = os.path.splitext(os.path.basename(config_path))[0]
    defaults = DEFAULTS.get(stem, {})

    data_root = args.data_root or defaults.get("data_root")
    ann_file  = args.ann_file  or defaults.get("ann_file")

    if data_root:
        cfg.test_dataloader.dataset.data_prefix.video = data_root
        src = "CLI" if args.data_root else "DEFAULTS"
        print(f"[data] data_prefix.video [{src}] → {data_root}")
    if ann_file:
        cfg.test_dataloader.dataset.ann_file = ann_file
        src = "CLI" if args.ann_file else "DEFAULTS"
        print(f"[data] ann_file          [{src}] → {ann_file}")

    # ── Checkpoint ───────────────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = args.checkpoint
        print(f"[ckpt] Using provided checkpoint:\n       {ckpt_path}")
    else:
        ckpt_path = find_best_checkpoint(config_path, PROJECT_ROOT)
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
    print(f"[ckpt] Checkpoint file size: {ckpt_size_mb:.2f} MB\n")

    # ── Build model ──────────────────────────────────────────────────────────
    model = init_recognizer(cfg, checkpoint=None, device=str(device))
    model = load_weights(model, ckpt_path, device)
    print()

    # ── Real / synthetic input ───────────────────────────────────────────────
    x_raw = get_input_tensor(cfg, device)
    print()

    # Pre-normalized tensor for the BackboneHeadWrapper
    # Extract single clip: (1, 3, 16, 224, 224)
    x_clip   = x_raw[:, 0]                       # (1, C, T, H, W)
    x_normed = normalize_input(x_clip, device)   # normalized in-place copy

    # ══════════════════════════════════════════════════════════════════════════
    #  TRAIN MODE
    # ══════════════════════════════════════════════════════════════════════════
    model.train()

    total_tr,     trainable_tr     = param_stats(model)
    fvcore_total_tr                = parameter_count(model)[""]   # '' = all params
    model_size_tr_mb               = total_tr * 4 / (1024 ** 2)

    diff_tr = abs(total_tr - fvcore_total_tr)

    print("----------------------------------")
    print("TRAIN MODE")
    print(f"Total Params (M):              {total_tr / 1e6:.4f}")
    print(f"Trainable Params (M):          {trainable_tr / 1e6:.4f}")
    print(f"Model Size from Params (MB):   {model_size_tr_mb:.2f}")
    print(f"Checkpoint File Size (MB):     {ckpt_size_mb:.2f}")
    if diff_tr == 0:
        print(f"  [fvcore cross-check] Matches manual count "
              f"({fvcore_total_tr / 1e6:.4f} M)  ✓")
    else:
        print(f"  [fvcore cross-check] {fvcore_total_tr / 1e6:.4f} M  "
              f"|  diff = {diff_tr:,} params  ← investigate!")

    # ══════════════════════════════════════════════════════════════════════════
    #  EVAL (TEST) MODE
    # ══════════════════════════════════════════════════════════════════════════
    model.eval()

    total_ev,     trainable_ev     = param_stats(model)
    fvcore_total_ev                = parameter_count(model)[""]
    model_size_ev_mb               = total_ev * 4 / (1024 ** 2)

    diff_ev = abs(total_ev - fvcore_total_ev)

    print("----------------------------------")
    print("EVAL MODE (TEST)")
    print(f"Total Params (M):              {total_ev / 1e6:.4f}")
    print(f"Trainable Params (M):          {trainable_ev / 1e6:.4f}")
    print(f"Model Size from Params (MB):   {model_size_ev_mb:.2f}")
    print(f"Checkpoint File Size (MB):     {ckpt_size_mb:.2f}")
    if diff_ev == 0:
        print(f"  [fvcore cross-check] Matches manual count "
              f"({fvcore_total_ev / 1e6:.4f} M)  ✓")
    else:
        print(f"  [fvcore cross-check] {fvcore_total_ev / 1e6:.4f} M  "
              f"|  diff = {diff_ev:,} params  ← investigate!")

    # ══════════════════════════════════════════════════════════════════════════
    #  EVAL MODE – SPEED & COMPUTE  (no gradients, CUDA-synced)
    # ══════════════════════════════════════════════════════════════════════════
    wrapper = BackboneHeadWrapper(model).to(device)
    wrapper.eval()

    with torch.no_grad():

        # ── Warm-up ───────────────────────────────────────────────────────────
        for _ in range(WARMUP_RUNS):
            _ = wrapper(x_normed)

        # ── Timed runs ────────────────────────────────────────────────────────
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(TIMED_RUNS):
            _ = wrapper(x_normed)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        mean_time_ms = (t1 - t0) / TIMED_RUNS * 1000.0

        # ── GFLOPs via fvcore ─────────────────────────────────────────────────
        flop_counter = FlopCountAnalysis(wrapper, x_normed)
        flop_counter.unsupported_ops_warnings(False)
        flop_counter.uncalled_modules_warnings(False)
        gflops = flop_counter.total() / 1e9

    print("----------------------------------")
    print("EVAL MODE (TEST) - SPEED & COMPUTE")
    print(f"Inference Time (ms):           {mean_time_ms:.2f}")
    print(f"GFLOPs:                        {gflops:.2f}")
    print("----------------------------------")


if __name__ == "__main__":
    main()
