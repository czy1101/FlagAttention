"""Host-side scheduling helpers shared by Parallax backward implementations."""

from contextlib import contextmanager, nullcontext

import torch


@contextmanager
def backward_streams(device, concurrent=True):
    """Join independent derivatives before returning to the caller's stream."""
    if not concurrent or torch.npu.is_current_stream_capturing():
        yield nullcontext(), nullcontext()
        return

    current = torch.npu.current_stream(device)
    dqr = torch.npu.Stream(device=device)
    dkv = torch.npu.Stream(device=device)
    dqr.wait_stream(current)
    dkv.wait_stream(current)
    try:
        yield torch.npu.stream(dqr), torch.npu.stream(dkv)
    finally:
        current.wait_stream(dqr)
        current.wait_stream(dkv)
