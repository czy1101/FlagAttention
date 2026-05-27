# Installing python3-flag-attention

After `apt install python3-flag-attention` (Debian/Ubuntu) or `dnf install python3-flag-attention` (Fedora),
install the ML runtime separately — it is intentionally **not** declared as a
hard Depends/Requires because the distro versions are CPU-only (torch) or
too old (triton) for GPU workloads.

FlagAttention needs Triton at runtime. Two options:

  - `apt install python3-flagtree-nvidia` (FlagOS-provided, satisfies the triton namespace), OR
  - `pip install triton` (matching your CUDA stack — distro python3-triton is too old)

Torch is also needed; install via pip matching your GPU stack:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0+cu128
```

A `venv` is recommended to isolate pip-installed packages from the system
Python:

```bash
python3 -m venv ~/.venv/flagos
source ~/.venv/flagos/bin/activate
# then the pip install lines above
```

Note: the distro `python3-torch` package (CPU-only build) is intentionally
not pulled in — FlagOS workloads need a GPU build, which PyTorch upstream
distributes via PyPI (per-CUDA-version wheels), not as a `.deb` / `.rpm`.
