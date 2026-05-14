"""
Measure params, model size, FLOPs, and inference latency for all SLR bench models.

Usage (GPU):
    python tools/analysis_tools/measure_model_complexity.py
    python tools/analysis_tools/measure_model_complexity.py --device cpu --warmup 5 --iters 20

Output:
    - Printed table to stdout
    - CSV saved to tools/analysis_tools/model_complexity.csv
"""

import argparse
import csv
import os
import time
import warnings

import torch
from mmengine import Config
from mmengine.registry import init_default_scope

warnings.filterwarnings("ignore")

# ── Model registry ─────────────────────────────────────────────────────────────
# One representative config per model family (wlasl100 train256, 100 classes).
# num_frames is read from each config automatically.
_BASE = "/home/zeynep/Thesis/code/mmaction2/configs/SLR"
#_BASE = "/arf/home/zgokce/code/mmaction2/configs/SLR"

MODELS = [
    # (display_name, config_relative_path)
    ("Swin-T",          f"{_BASE}/bench_swin_tiny/swin_tiny_wlasl100_train256.py"),
    ("Swin-S",          f"{_BASE}/bench_swin_small/swin_small_wlasl100_train256.py"),
    ("Swin-B",          f"{_BASE}/bench_swin_base/swin_base_wlasl100_train256.py"),
    ("Swin-L",          f"{_BASE}/bench_swin_large/swin_large_wlasl100_train256.py"),
    ("UniFormer-S",     f"{_BASE}/bench_uniformer_small/uniformer_small_wlasl100_train256.py"),
    ("UniFormer-B",     f"{_BASE}/bench_uniformer_base/uniformer_base_wlasl100_train256.py"),
    ("UniFormerV2-B",   f"{_BASE}/bench_uniformer_v2_base/uniformer_v2_base_wlasl100_train256.py"),
    ("UniFormerV2-L",   f"{_BASE}/bench_uniformer_v2_large/uniformer_v2_large_wlasl100_train256.py"),
    ("VideoMAEv2-S",    f"{_BASE}/bench_videomaev2_small/videomaev2_small_wlasl100_train256.py"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Measure model complexity for all SLR bench models")
    p.add_argument("--device",  default="cuda:0", help="cuda:0 or cpu")
    p.add_argument("--warmup",  type=int, default=50, help="warmup iterations for timing")
    p.add_argument("--iters",   type=int, default=100, help="timed iterations")
    p.add_argument("--out",     default="./model_complexity.csv",
                   help="output CSV path")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total


def model_size_mb(model, dtype_bytes=4):
    return count_params(model) * dtype_bytes / (1024 ** 2)


def measure_flops(model, input_shape):
    """Return (flops_str, params_str) using mmengine analysis."""
    try:
        from mmengine.analysis import get_model_complexity_info
    except ImportError:
        return "N/A", "N/A"

    orig_forward = model.forward
    if hasattr(model, "extract_feat"):
        model.forward = model.extract_feat

    try:
        result = get_model_complexity_info(model, input_shape)
        flops_str  = result["flops_str"]
        params_str = result["params_str"]
    except Exception as e:
        flops_str  = f"ERR({e.__class__.__name__})"
        params_str = "N/A"
    finally:
        model.forward = orig_forward

    return flops_str, params_str


@torch.no_grad()
def measure_latency(model, input_shape, device, warmup, iters):
    x = torch.randn(*input_shape, device=device)

    # use backbone directly — extract_feat may expect 6D input (N, M, C, T, H, W)
    forward_fn = model.backbone if hasattr(model, "backbone") else model

    # warmup
    for _ in range(warmup):
        forward_fn(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward_fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    p50_ms = sorted(times)[len(times) // 2] * 1000
    return avg_ms, p50_ms


def human_params(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    return f"{n/1e3:.1f}K"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device)

    rows = []  # list of dicts for CSV

    header = f"{'Model':<18} {'Params':>8} {'Size(MB)':>9} {'GFLOPs':>10} " \
             f"{'Avg(ms)':>9} {'P50(ms)':>9} {'Frames':>7}"
    sep    = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for display_name, cfg_path in MODELS:
        print(f"  → Loading {display_name} ...", flush=True)

        # ── build model ─────────────────────────────────────────────────────
        try:
            cfg = Config.fromfile(cfg_path)
        except Exception as e:
            print(f"    [SKIP] Config error: {e}")
            continue

        init_default_scope(cfg.get("default_scope", "mmaction"))

        # patch num_classes to 100 in case config has a different value
        cfg.model.cls_head.num_classes = 100

        try:
            from mmaction.registry import MODELS as MM_MODELS
            model = MM_MODELS.build(cfg.model).eval().to(device)
        except Exception as e:
            print(f"    [SKIP] Build error: {e}")
            continue

        # ── num_frames from config ───────────────────────────────────────────
        num_frames = cfg.get("num_frames", 16)
        input_shape = (1, 3, num_frames, 224, 224)

        # ── params & size ───────────────────────────────────────────────────
        n_params = count_params(model)
        size_mb  = model_size_mb(model)

        # ── FLOPs ────────────────────────────────────────────────────────────
        flops_str, _ = measure_flops(model, input_shape)
        # convert to GFLOPs float — handle both "140.96 GFLOPs" and "0.14096T"
        try:
            parts = flops_str.split()
            if len(parts) == 2:
                val, unit = parts
                multipliers = {"GFLOPs": 1, "G": 1, "TFLOPs": 1000, "T": 1000,
                               "MFLOPs": 0.001, "M": 0.001}
                gflops_val = float(val) * multipliers.get(unit, 1)
            else:
                s = parts[0]
                for suffix, mult in [("T", 1000), ("G", 1), ("M", 0.001)]:
                    if s.upper().endswith(suffix):
                        gflops_val = float(s[:-1]) * mult
                        break
                else:
                    raise ValueError(f"unrecognised format: {flops_str}")
            gflops_display = f"{gflops_val:.1f}G"
        except Exception:
            gflops_display = flops_str

        # ── latency ──────────────────────────────────────────────────────────
        try:
            avg_ms, p50_ms = measure_latency(
                model, input_shape, device, args.warmup, args.iters)
            avg_str = f"{avg_ms:.1f}"
            p50_str = f"{p50_ms:.1f}"
        except Exception as e:
            avg_str = p50_str = f"ERR"
            avg_ms  = p50_ms  = -1.0
            print(f"    [WARN] Latency error: {e}")

        # ── print row ────────────────────────────────────────────────────────
        row = (f"{display_name:<18} {human_params(n_params):>8} "
               f"{size_mb:>9.1f} {gflops_display:>10} "
               f"{avg_str:>9} {p50_str:>9} {num_frames:>7}")
        print(row)

        rows.append({
            "model":          display_name,
            "params":         n_params,
            "params_human":   human_params(n_params),
            "size_mb":        round(size_mb, 2),
            "flops":          gflops_display,
            "avg_latency_ms": round(avg_ms, 2),
            "p50_latency_ms": round(p50_ms, 2),
            "num_frames":     num_frames,
            "device":         str(device),
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"{sep}\n")

    # ── CSV output ─────────────────────────────────────────────────────────────
    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved → {args.out}\n")


if __name__ == "__main__":
    main()
