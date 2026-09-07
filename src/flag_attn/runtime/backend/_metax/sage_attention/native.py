# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional EXP10A inference path; unsupported inputs retain Triton behavior."""

import os

import torch

from . import _native_loader
from .attn_qk_int8_per_block import forward as triton_forward


def unsupported_reason(q, k, v, q_scale, k_scale, tensor_layout="HND", attn_mask=None,
                       output_dtype=torch.float16, return_lse=False, maxnreg=None):
    if tensor_layout != "HND":
        return "layout"
    if attn_mask is not None or return_lse or maxnreg is not None:
        return "mask_lse_or_maxnreg"
    tensors = (q, k, v, q_scale, k_scale)
    if any(tensor.__class__ is not torch.Tensor for tensor in tensors):
        return "tensor_type"
    if any(not tensor.is_cuda or tensor.device != q.device for tensor in tensors):
        return "device"
    if (q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.float16
            or output_dtype != torch.float16 or q_scale.dtype != torch.float32
            or k_scale.dtype != torch.float32):
        return "dtype"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return "rank"
    b, h, nq, d = q.shape
    if d != 128 or k.shape[-1] != 128:
        return "head_dim"
    if tuple(k.shape[:2]) != (b, h) or v.shape != k.shape:
        return "batch_or_heads"
    nk = k.shape[2]
    if b <= 0 or h <= 0 or nq <= 0 or nk <= 0 or nq % 128 or nk % 64:
        return "length"
    int_max = (1 << 31) - 1
    if max(b * h, nq, nk, b * h * (nq // 128)) > int_max:
        return "grid_range"
    if tuple(q_scale.shape) != (b, h, nq // 128) or tuple(k_scale.shape) != (b, h, nk // 64):
        return "scale_shape"
    if any(not tensor.is_contiguous() for tensor in tensors):
        return "strides"
    if any(tensor.data_ptr() == 0 or tensor.data_ptr() % alignment
           for tensor, alignment in zip(tensors, (16, 16, 16, 4, 4))):
        return "alignment"
    if any(tensor.requires_grad for tensor in tensors):
        return "autograd"
    if torch.cuda.get_device_name(q.device) != "MetaX C550":
        return "hardware"
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            return "graph_capture"
    return None


def _empty_output(q):
    return torch.empty(q.shape, dtype=torch.float16, device=q.device)


def forward(q, k, v, q_scale, k_scale, tensor_layout="HND", attn_mask=None,
            output_dtype=torch.float16, return_lse=False, maxnreg=None):
    arguments = (q, k, v, q_scale, k_scale, tensor_layout, attn_mask,
                 output_dtype, return_lse, maxnreg)
    directory = os.environ.get("FLAG_ATTN_SAGE_NATIVE_DIR", "")
    if not directory or unsupported_reason(*arguments) is not None:
        return triton_forward(*arguments)
    with torch.cuda.device(q.device):
        extension = _native_loader.load_extension(directory, torch.__version__)
        if extension is None:
            return triton_forward(*arguments)
        stream = torch.cuda.current_stream(q.device)
        out = _empty_output(q)
        # Readiness is the caller's responsibility, as for other Torch ops.
        # record_stream protects allocator lifetime during asynchronous use.
        for tensor in (q, k, v, q_scale, k_scale, out):
            tensor.record_stream(stream)
        b, h, nq, _ = q.shape
        extension.launch(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                         q_scale.data_ptr(), k_scale.data_ptr(), out.data_ptr(),
                         b * h, nq, k.shape[2], stream.cuda_stream)
    # Never catch a native launch error and silently rerun Triton.
    return out, torch.empty([0], dtype=torch.float32, device="cpu")
