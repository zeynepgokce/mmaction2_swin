# get_infer_time.py
import argparse
import time
import torch

from mmengine import Config
from mmengine.registry import init_default_scope
from mmaction.registry import MODELS

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def model_size_mb(model, dtype_bytes=4):
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * dtype_bytes / (1024 ** 2)
    return size_mb


def parse_args():
    p = argparse.ArgumentParser("Measure inference latency for MMAction2 model")
    p.add_argument("config", help="config file path")
    p.add_argument("--shape", type=int, nargs="+", required=True,
                   help="Input shape. e.g. 1 3 32 224 224 (N C T H W)")
    p.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    p.add_argument("--iters", type=int, default=100, help="timed iterations")
    p.add_argument("--warmup", type=int, default=50, help="warmup iterations")
    p.add_argument("--fp16", action="store_true", help="use autocast fp16 on cuda")
    p.add_argument("--use_extract_feat", action="store_true",
                   help="measure extract_feat instead of full forward")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    input_shape = tuple(args.shape)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmaction"))
    model = MODELS.build(cfg.model).eval()

    device = torch.device(args.device)
    model = model.to(device)

    if args.use_extract_feat and hasattr(model, "extract_feat"):
        model.forward = model.extract_feat

    # dummy input
    x = torch.randn(*input_shape, device=device)

    # warmup
    for _ in range(args.warmup):
        if device.type == "cuda" and args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(x)
        else:
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # timed
    times = []
    for _ in range(args.iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if device.type == "cuda" and args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(x)
        else:
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times)//2]
    fps = 1.0 / avg if avg > 0 else float("inf")

    print(f"Input shape: {input_shape}  Device: {device}")
    print(f"Avg latency: {avg*1000:.3f} ms")
    print(f"P50 latency: {p50*1000:.3f} ms")
    print(f"FPS (approx): {fps:.2f}")

    total, trainable = count_params(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")

    size_mb = model_size_mb(model)
    print(f"Model size (FP32): {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
