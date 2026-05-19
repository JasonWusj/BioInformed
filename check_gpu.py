"""
测试 PyTorch 版本和 CUDA 可用性。

如果显示 CUDA not available，请在项目目录下运行：
  .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
"""
import sys

try:
    import torch
except ImportError:
    print("ERROR: torch not installed")
    sys.exit(1)

# 基本信息
version = getattr(torch, '__version__', 'unknown')
cuda_version = getattr(torch.version, 'cuda', None)
print(f"PyTorch: {version}")
print(f"CUDA compiled: {cuda_version or 'None (CPU-only build!)'}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"VRAM: {vram:.1f} GB")
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x.T
    print(f"GPU compute test: OK")
else:
    print()
    print("=" * 50)
    print("WARNING: 当前是 CPU 版本的 PyTorch!")
    print("=" * 50)
    print("修复方法（在项目目录下执行）：")
    print()
    print("  .venv\\Scripts\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall")
    print()
    print("注意：CUDA 版 torch 约 2.5GB，下载需要一些时间。")
