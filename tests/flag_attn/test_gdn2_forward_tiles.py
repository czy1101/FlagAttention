"""GDN2-003: adaptive per-call S60 tile sizes for the fused Enflame forward.

The fused GDN-2 inference kernels used a fixed BK=BV=256 tile even when K or V
was only 64, so most lanes of every dot were masked.  ``_s60_tile`` clamps the
tile to the real extent (never below 64, never above the tuned 256).  These
tests pin the tile map, show that the clamp changes only the launch geometry,
and compare the adaptive launch with the legacy fixed-256 launch on device.

Nothing here exercises a backward pass: ``chunk_gdn2`` is inference-only.
"""

import importlib
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

gdn2_impl = importlib.import_module(
    "flag_attn.runtime.backend._enflame.FLA.gdn2.chunk_gdn2"
)
from flag_attn.testing import backend as test_backend  # noqa: E402

pytestmark = pytest.mark.skipif(
    not test_backend.supports_operator("chunk_gdn2"),
    reason="chunk_gdn2 is unavailable on the active backend",
)

TILE_CASES = [
    # (extent, limit, expected tile)
    (1, 256, 64), (32, 256, 64), (64, 256, 64), (65, 256, 128),
    (128, 256, 128), (129, 256, 256), (255, 256, 256), (256, 256, 256),
    (64, 128, 64), (64, 64, 64), (256, 128, 128),
]


@pytest.mark.parametrize("extent,limit,expected", TILE_CASES)
def test_s60_tile_is_power_of_two_and_covers_extent(extent, limit, expected):
    tile = gdn2_impl._s60_tile(extent, limit)
    assert tile == expected
    assert 64 <= tile <= 256
    assert tile & (tile - 1) == 0, "tile must stay a power of two"
    assert tile >= min(extent, limit), "tile must cover the logical extent"


def test_s60_tile_respects_limit_and_legacy_switch(monkeypatch):
    assert gdn2_impl._s60_tile(256, 128) == 128
    monkeypatch.setattr(gdn2_impl, "_S60_TILES_ADAPT", False)
    assert gdn2_impl._s60_tile(64, 256) == 256, "legacy switch must restore the fixed tile"


def _make(B, T, H, K, V, dtype, device):
    torch.manual_seed(42)
    q = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
    k = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
    v = torch.randn(B, T, H, V).to(dtype)
    g = (-torch.rand(B, T, H, K) * 0.1).to(dtype)
    b = torch.rand(B, T, H, K).to(dtype)
    w = torch.rand(B, T, H, V).to(dtype)
    h0 = torch.randn(B, H, K, V) * 0.01
    return [x.to(device) for x in (q, k, v, g, b, w, h0)]


def _capture_launches(monkeypatch):
    from triton.runtime.jit import JITFunction

    seen = []
    original = JITFunction.run

    def traced(self, *args, **kwargs):
        seen.append({
            "kernel": getattr(self.fn, "__name__", str(self.fn)),
            "BK": kwargs.get("BK"),
            "BV": kwargs.get("BV"),
            "BT": kwargs.get("BT"),
        })
        return original(self, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "run", traced)
    return seen


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize(
    "shape,tiles",
    [
        ((2, 130, 2, 64, 64), {"intra": (64, 64), "state": (64, 64)}),
        ((2, 130, 2, 128, 128), {"intra": (128, 128), "state": (128, 128)}),
        ((1, 130, 1, 256, 256), {"intra": (256, 256), "state": (256, 256)}),
    ],
    ids=["k64v64", "k128v128", "k256v256"],
)
@torch.inference_mode()
def test_adaptive_tiles_are_launched_and_output_is_unchanged(shape, tiles, monkeypatch):
    from flag_attn import chunk_gdn2

    device = test_backend.device
    B, T, H, K, V = shape
    args = _make(B, T, H, K, V, torch.bfloat16, device)
    scale = K ** -0.5
    kwargs = dict(scale=scale, initial_state=args[6], output_final_state=True, chunk_size=64)

    monkeypatch.setattr(gdn2_impl, "_S60_TILES_ADAPT", True)
    seen = _capture_launches(monkeypatch)
    o_new, ht_new = chunk_gdn2(*args[:6], **kwargs)
    torch.gcu.synchronize()
    assert seen, "no Triton kernel was launched"
    by_name = {item["kernel"]: item for item in seen}
    intra = by_name["_chunk_gdn2_fwd_intra_infer_kernel"]
    state = by_name["_chunk_gdn2_fwd_h_o_infer_kernel"]
    assert (intra["BK"], intra["BV"]) == tiles["intra"]
    assert (state["BK"], state["BV"]) == tiles["state"]

    monkeypatch.setattr(gdn2_impl, "_S60_TILES_ADAPT", False)
    o_ref, ht_ref = chunk_gdn2(*args[:6], **kwargs)
    torch.gcu.synchronize()

    assert torch.equal(o_new, o_ref), "adaptive tile changed the output"
    assert torch.equal(ht_new, ht_ref), "adaptive tile changed the final state"
    assert torch.isfinite(o_new).all() and torch.isfinite(ht_new).all()
