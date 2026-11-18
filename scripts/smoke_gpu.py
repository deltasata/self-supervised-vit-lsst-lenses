"""1-GPU smoke test: environment, GPU visibility, memory, allocation size, a quick matmul/conv benchmark."""
import os
import subprocess
import time

import torch

print("hostname", os.uname().nodename)
for k in ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
          "CUDA_VISIBLE_DEVICES", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "OMP_NUM_THREADS"):
    print(f"{k}={os.environ.get(k)}")
print("torch", torch.__version__, "hip", torch.version.hip, "cuda.is_available", torch.cuda.is_available())
n = torch.cuda.device_count()
print("device_count", n)
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, f"{p.total_memory / 2**30:.1f} GiB", "gcnArch", getattr(p, "gcnArchName", "?"),
          "CUs", p.multi_processor_count)
print("os.cpu_count", os.cpu_count(), "sched_getaffinity", len(os.sched_getaffinity(0)))
for cmd in (["rocm-smi", "--showproductname", "--showmeminfo", "vram"], ["rocm-smi"]):
    try:
        print(subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout)
    except Exception as e:
        print("rocm-smi failed", e)

dev = "cuda"
for dt in (torch.float32, torch.bfloat16):
    a = torch.randn(8192, 8192, device=dev, dtype=dt)
    b = torch.randn(8192, 8192, device=dev, dtype=dt)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    dtm = time.time() - t
    print(f"matmul {dt}: {20 * 2 * 8192**3 / dtm / 1e12:.1f} TFLOP/s")

import timm  # noqa: E402
m = timm.create_model("resnet18", num_classes=1, in_chans=5).to(dev).to(memory_format=torch.channels_last)
x = torch.randn(512, 5, 64, 64, device=dev).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(m.parameters())
for i in range(25):
    if i == 5:
        torch.cuda.synchronize()
        t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(x).float().mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
torch.cuda.synchronize()
print(f"resnet18 5ch 64px bs512 bf16 train: {20 * 512 / (time.time() - t):.0f} img/s")
q = torch.randn(64, 6, 1024, 64, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize()
t = time.time()
for _ in range(20):
    o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
torch.cuda.synchronize()
print(f"SDPA ok, {(time.time() - t) / 20 * 1e3:.1f} ms/call")
print("max_memory_allocated GiB", torch.cuda.max_memory_allocated() / 2**30)
print("SMOKE OK")
