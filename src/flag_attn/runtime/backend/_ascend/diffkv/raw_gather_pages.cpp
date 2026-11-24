// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

static __aicore__ __attribute__((always_inline)) void copy_rows(
    __cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, int rows, int width,
    int row_stride, int tile_tokens) {
  copy_gm_to_cbuf_multi_nd2nz_b16(dst, src, 0, 1, rows, width, 0, row_stride,
                                tile_tokens, 1, 0);
}

extern "C" __aicore__ void _mlir_ciface_diffkv_gather_pages_bf16(
    memref_t<__gm__ bfloat16_t, 2> *source,
    memref_t<__gm__ int32_t, 2> *pages,
    memref_t<__gm__ bfloat16_t, 2> *zeros,
    int64_t tile_tokens, int64_t width, int64_t page_stride, int64_t row_stride,
    int64_t first_valid, int64_t valid_end,
    memref_t<__cbuf__ bfloat16_t, 4> *dest) {
  auto source_base = source->aligned + source->offset;
  auto page_ids = pages->aligned + pages->offset;
  auto zero_base = zeros->aligned + zeros->offset;
  auto tile = dest->aligned + dest->offset;
  constexpr int page_tokens = 16;
  constexpr int lanes = 16;

  // Each token offset advances one C0 block in the NZ tile.
  // Fill invalid rows explicitly so padded cache contents cannot enter PV.
  for (int i = 0; i < tile_tokens / page_tokens; ++i) {
    int begin = i * page_tokens;
    int lo = first_valid > begin ? first_valid - begin : 0;
    lo = lo < page_tokens ? lo : page_tokens;
    int hi = valid_end - begin;
    hi = hi > lo ? hi : lo;
    hi = hi < page_tokens ? hi : page_tokens;
    auto dst = tile + begin * lanes;
    if (lo > 0)
      copy_rows(dst, zero_base, lo, width, 192, tile_tokens);
    if (hi > lo) {
      auto page = static_cast<int64_t>(page_ids[i * pages->strides[0]]);
      copy_rows(dst + lo * lanes, source_base + page * page_stride + lo * row_stride,
                hi - lo, width, row_stride, tile_tokens);
    }
    if (hi < page_tokens)
      copy_rows(dst + hi * lanes, zero_base, page_tokens - hi, width, 192, tile_tokens);
  }
}
