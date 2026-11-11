import pytest
import torch
import torch_gcu  # noqa: F401

torch.gcu.set_device(1)
raise SystemExit(
    pytest.main(["-q", "-s", "tests/flag_attn/test_chunk_gla.py", "-k", "B1-T63-H1-D64"])
)
