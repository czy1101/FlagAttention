#include <stdint.h>

static __device__ __forceinline__ float decode_exp2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

static __device__ __forceinline__ uint16_t decode_float_to_bfloat16_bits(float x) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = x;
  const uint32_t lsb = (v.u >> 16) & 1u;
  return static_cast<uint16_t>((v.u + 0x7fffu + lsb) >> 16);
}

static __device__ __forceinline__ float decode_warp_reduce_max(float v) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, v, mask);
    v = other > v ? other : v;
  }
  return v;
}

static __device__ __forceinline__ float decode_warp_reduce_sum(float v) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, mask);
  }
  return v;
}

static __device__ __forceinline__ uint32_t decode_arrive_acq_rel(
    __attribute__((address_space(1))) int32_t *counter) {
  uint32_t old;
  const uint32_t one = 1;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(counter), "r"(one)
               : "memory");
  return old;
}

static __device__ __forceinline__ void decode_reset_release(
    __attribute__((address_space(1))) int32_t *counter) {
  uint32_t old;
  const uint32_t zero = 0;
  asm volatile("atom.release.gpu.global.exch.b32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(counter), "r"(zero)
               : "memory");
}

__device__ void DynamicSplitKFinalize(
    __attribute__((address_space(1))) const float *split_out,
    __attribute__((address_space(1))) const float *lse,
    __attribute__((address_space(1))) int32_t *completion,
    __attribute__((address_space(1))) volatile int32_t *last_flags,
    __attribute__((address_space(1))) uint16_t *out,
    const int hkv, const int batch, const int n_chunks, const int batch_count,
    const int H_Q, const int HEADS_PER_GROUP, const int DV,
    const int64_t SO_STRIDE_B, const int64_t SO_STRIDE_C,
    const int64_t SO_STRIDE_M, const int64_t SO_STRIDE_H,
    const int64_t LSE_STRIDE_B, const int64_t LSE_STRIDE_C,
    const int64_t LSE_STRIDE_HKV, const int64_t LSE_STRIDE_M,
    const int64_t LSE_STRIDE_HG, const int64_t O_STRIDE_B,
    const int64_t O_STRIDE_M, const int64_t O_STRIDE_H) {
  constexpr int kWarps = 4;
  constexpr int kItemsPerLane = 4;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int counter_idx = hkv * batch_count + batch;

  // The CTA barrier publishes every lane's split_out/LSE stores to thread 0.
  // Its release RMW then makes the complete partial visible to the final CTA.
  __syncthreads();
  if (tid == 0) {
    const uint32_t ticket = decode_arrive_acq_rel(completion + counter_idx);
    last_flags[blockIdx.x] = ticket == static_cast<uint32_t>(n_chunks - 1);
  }
  __syncthreads();

  if (!last_flags[blockIdx.x]) {
    return;
  }

#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int h_in_group = warp + pass * kWarps;
    const int hq = hkv * HEADS_PER_GROUP + h_in_group;
    if (h_in_group < HEADS_PER_GROUP && hq < H_Q) {
      float lane_max = -__builtin_inff();
      for (int c = lane; c < n_chunks; c += 32) {
        const float value = lse[batch * LSE_STRIDE_B + c * LSE_STRIDE_C +
                                hkv * LSE_STRIDE_HKV + 0 * LSE_STRIDE_M +
                                h_in_group * LSE_STRIDE_HG];
        lane_max = value > lane_max ? value : lane_max;
      }
      const float max_lse = decode_warp_reduce_max(lane_max);
      const float safe_max_lse = max_lse == -__builtin_inff() ? 0.0f : max_lse;

      float lane_denom = 0.0f;
      for (int c = lane; c < n_chunks; c += 32) {
        const float value = lse[batch * LSE_STRIDE_B + c * LSE_STRIDE_C +
                                hkv * LSE_STRIDE_HKV + 0 * LSE_STRIDE_M +
                                h_in_group * LSE_STRIDE_HG];
        const float delta = value - safe_max_lse;
        lane_denom += decode_exp2_approx(delta);
      }
      const float denom = decode_warp_reduce_sum(lane_denom);
      const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;

      float acc[kItemsPerLane];
#pragma unroll
      for (int i = 0; i < kItemsPerLane; ++i) {
        acc[i] = 0.0f;
      }

      const int v_base = lane * kItemsPerLane;
      for (int c = 0; c < n_chunks; ++c) {
        float chunk_lse = 0.0f;
        if (lane == 0) {
          chunk_lse = lse[batch * LSE_STRIDE_B + c * LSE_STRIDE_C +
                          hkv * LSE_STRIDE_HKV + 0 * LSE_STRIDE_M +
                          h_in_group * LSE_STRIDE_HG];
        }
        chunk_lse = __shfl_sync(0xffffffffu, chunk_lse, 0);
        const float delta = chunk_lse - safe_max_lse;
        const float weight = decode_exp2_approx(delta) * inv_denom;

#pragma unroll
        for (int i = 0; i < kItemsPerLane; ++i) {
          const int v = v_base + i;
          if (v < DV) {
            const float value =
                split_out[batch * SO_STRIDE_B + c * SO_STRIDE_C +
                          0 * SO_STRIDE_M + hq * SO_STRIDE_H + v];
            acc[i] += weight * value;
          }
        }
      }

#pragma unroll
      for (int i = 0; i < kItemsPerLane; ++i) {
        const int v = v_base + i;
        if (v < DV) {
          const float value = denom > 0.0f ? acc[i] : 0.0f;
          out[batch * O_STRIDE_B + 0 * O_STRIDE_M + hq * O_STRIDE_H + v] =
              decode_float_to_bfloat16_bits(value);
        }
      }
    }
  }

  __syncthreads();
  if (tid == 0) {
    decode_reset_release(completion + counter_idx);
  }
}
