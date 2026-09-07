// Copyright 2026 FlagAttention Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <mctlass/arch/mma_sm80.h>
#include <mctlass/half.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using I8Mma = mctlass::arch::Mma<
    mctlass::gemm::GemmShape<16, 16, 16>,
    32,
    int8_t,
    mctlass::layout::RowMajor,
    int8_t,
    mctlass::layout::ColumnMajor,
    int,
    mctlass::layout::RowMajor,
    mctlass::arch::OpMultiplyAddSaturate>;

using F16Mma = mctlass::arch::Mma<
    mctlass::gemm::GemmShape<16, 16, 16>,
    32,
    mctlass::half_t,
    mctlass::layout::RowMajor,
    mctlass::half_t,
    mctlass::layout::ColumnMajor,
    float,
    mctlass::layout::RowMajor,
    mctlass::arch::OpMultiplyAdd>;

constexpr int CTA_M = 128;
constexpr int CTA_KV = 64;
constexpr int KV_BLOCKS = 2;
constexpr int TOTAL_KV = CTA_KV * KV_BLOCKS;
constexpr int DHEAD = 128;
constexpr int OUT_D = 128;
constexpr int THREADS = 256;
constexpr int LANES = 64;
constexpr int WAVES = 4;
constexpr int MMA_M = 2;
constexpr int QK_MMA_N = 4;
constexpr int QK_K_TILES = 8;
constexpr int PV_N_TILES = 8;
constexpr int PV_K_TILES = 4;
constexpr int Q_ELEMENTS = CTA_M * DHEAD;
constexpr int K_BLOCK_ELEMENTS = CTA_KV * DHEAD;
constexpr int V_BLOCK_ELEMENTS = CTA_KV * OUT_D;
constexpr int V_BLOCK_BYTES =
    V_BLOCK_ELEMENTS * static_cast<int>(sizeof(mctlass::half_t));
constexpr int K_ELEMENTS = TOTAL_KV * DHEAD;
constexpr int V_ELEMENTS = TOTAL_KV * OUT_D;
constexpr int QK_ELEMENTS = CTA_M * TOTAL_KV;
constexpr int O_ELEMENTS = CTA_M * OUT_D;
constexpr int REPEATS = 20;
constexpr int CASES = 4;
constexpr int PERF_Q_TILES_PER_HEAD = 2;
constexpr int PERF_MAX_TASKS = 512;
constexpr int PERF_MAX_HEADS =
    (PERF_MAX_TASKS + PERF_Q_TILES_PER_HEAD - 1) /
    PERF_Q_TILES_PER_HEAD;
constexpr int PERF_WARMUP = 25;
constexpr int PERF_REPEATS = 100;
constexpr int PERF_TRIALS = 5;
constexpr int K_SWIZZLE_WARMUP = 25;
constexpr int K_SWIZZLE_TRIALS = 15;
constexpr double K_SWIZZLE_TARGET_TIMING_MS = 60.0;
constexpr int K_SWIZZLE_MIN_REPEATS = 5;
constexpr int K_SWIZZLE_MAX_REPEATS = 2000;
constexpr int GUARD = 32;
constexpr int32_t I32_SENTINEL = 0x5a5a5a5a;
constexpr float FLOAT_SENTINEL = 1234567.0f;
constexpr int32_t OWNER_SENTINEL = -7777777;
constexpr unsigned long long FULL_WAVE_MASK =
    0xffffffffffffffffULL;

static_assert(sizeof(I8Mma::FragmentA) == 4, "bad I8 FragmentA");
static_assert(sizeof(I8Mma::FragmentB) == 4, "bad I8 FragmentB");
static_assert(sizeof(I8Mma::FragmentC) == 16, "bad I8 FragmentC");
static_assert(sizeof(F16Mma::FragmentA) == 8, "bad F16 FragmentA");
static_assert(sizeof(F16Mma::FragmentB) == 8, "bad F16 FragmentB");
static_assert(sizeof(F16Mma::FragmentC) == 16, "bad F16 FragmentC");
static_assert(sizeof(mctlass::half_t) == 2, "half_t must be 16-bit");
static_assert(sizeof(uint4) == 16, "uint4 must be 16 bytes");
static_assert(V_BLOCK_BYTES % static_cast<int>(sizeof(uint4)) == 0,
              "V block must be uint4 aligned");
static_assert(QK_MMA_N == PV_K_TILES, "QK/PV K tile mismatch");
static_assert(DHEAD == OUT_D, "routing harness reuses K/V element counts");
static_assert(WAVES * MMA_M * 16 == CTA_M, "bad CTA M coverage");
static_assert(MMA_M == 2, "CTA128 path requires two M tiles per wave");

template <bool STAGE_V, bool SWIZZLE_K = false>
__global__ __launch_bounds__(THREADS, 1)
void qk_softmax_pv_dynamic_kv_kernel(
    const int8_t *Q,
    const int8_t *K,
    const mctlass::half_t *V,
    const float *Q_scale,
    const float *K_scale,
    mctlass::half_t *O,
    int q_len,
    int kv_len,
    int q_tiles,
    int q_scale_blocks,
    int kv_blocks) {
  __shared__ __align__(16) int8_t shared_k[K_BLOCK_ELEMENTS];
  extern __shared__ __align__(16) unsigned char shared_v_storage[];
  mctlass::half_t *shared_v =
      reinterpret_cast<mctlass::half_t *>(shared_v_storage);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const int row = lane & 15;
  const int group = lane >> 4;
  const int task = static_cast<int>(blockIdx.x);
  const int head = task / q_tiles;
  const int q_tile = task - head * q_tiles;
  const int q_start = q_tile * CTA_M;

  const int8_t *q_head = Q + static_cast<size_t>(head) * q_len * DHEAD;
  const int8_t *k_head = K + static_cast<size_t>(head) * kv_len * DHEAD;
  const mctlass::half_t *v_head =
      V + static_cast<size_t>(head) * kv_len * OUT_D;
  mctlass::half_t *o_head = O + static_cast<size_t>(head) * q_len * OUT_D;
  const float q_scale =
      Q_scale[static_cast<size_t>(head) * q_scale_blocks +
              q_start / 128];

  I8Mma::FragmentA resident_q[MMA_M][QK_K_TILES];

#pragma unroll
  for (int tile = 0; tile < QK_K_TILES; ++tile) {
    const int k0 = tile * 16 + group * 4;
#pragma unroll
    for (int p = 0; p < MMA_M; ++p) {
      const int m = (wave * MMA_M + p) * 16 + row;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        resident_q[p][tile][i] =
            q_head[
                static_cast<size_t>(q_start + m) * DHEAD +
                k0 + i];
      }
    }
  }

  F16Mma::FragmentC numerator_acc[MMA_M][PV_N_TILES];
  float running_m[MMA_M];
  float running_l[MMA_M];

#pragma unroll
  for (int p = 0; p < MMA_M; ++p) {
    running_m[p] = -INFINITY;
    running_l[p] = 0.0f;
  }

#pragma unroll
  for (int p = 0; p < MMA_M; ++p) {
#pragma unroll
    for (int db = 0; db < PV_N_TILES; ++db) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        numerator_acc[p][db][i] = 0.0f;
      }
    }
  }

#pragma unroll 1
  for (int kv_block = 0; kv_block < kv_blocks; ++kv_block) {
    if (kv_block != 0) {
      __syncthreads();
    }

#pragma unroll
    for (int vector_index = tid;
         vector_index < K_BLOCK_ELEMENTS / 16;
         vector_index += THREADS) {
      if constexpr (SWIZZLE_K) {
        constexpr int vectors_per_row = DHEAD / 16;
        const int key = vector_index / vectors_per_row;
        const int logical_vector =
            vector_index - key * vectors_per_row;
        const int row_pair = (key >> 1) & (vectors_per_row - 1);
        const int physical_vector = logical_vector ^ row_pair;
        reinterpret_cast<uint4 *>(shared_k)
            [key * vectors_per_row + physical_vector] =
            reinterpret_cast<const uint4 *>(
                k_head + static_cast<size_t>(kv_block) *
                             K_BLOCK_ELEMENTS)[vector_index];
      } else {
        reinterpret_cast<uint4 *>(shared_k)[vector_index] =
            reinterpret_cast<const uint4 *>(
                k_head + static_cast<size_t>(kv_block) *
                             K_BLOCK_ELEMENTS)[vector_index];
      }
    }

    if constexpr (STAGE_V) {
#pragma unroll
      for (int vector_index = tid;
           vector_index < V_BLOCK_BYTES / 16;
           vector_index += THREADS) {
        constexpr int vectors_per_row = OUT_D / 8;
        const int key = vector_index / vectors_per_row;
        const int d_vector = vector_index - key * vectors_per_row;
        const int swizzle = ((((key >> 2) & 3) << 1) ^ (key & 3));
        const int physical_vector = d_vector ^ swizzle;
        reinterpret_cast<uint4 *>(shared_v)
            [key * vectors_per_row + physical_vector] =
            reinterpret_cast<const uint4 *>(
                v_head + static_cast<size_t>(kv_block) *
                             V_BLOCK_ELEMENTS)[vector_index];
      }
    }

    __syncthreads();

    F16Mma::FragmentA p_fragment[MMA_M][PV_K_TILES];
    float row_alpha[MMA_M];

    {
      I8Mma::FragmentC qk_acc[MMA_M][QK_MMA_N];

#pragma unroll
      for (int p = 0; p < MMA_M; ++p) {
#pragma unroll
        for (int nb = 0; nb < QK_MMA_N; ++nb) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            qk_acc[p][nb][i] = 0;
          }
        }
      }

#pragma unroll
      for (int tile = 0; tile < QK_K_TILES; ++tile) {
        I8Mma::FragmentB b[QK_MMA_N];
        const int k0 = tile * 16 + group * 4;

#pragma unroll
        for (int nb = 0; nb < QK_MMA_N; ++nb) {
          const int n = nb * 16 + row;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            if constexpr (SWIZZLE_K) {
              constexpr int vectors_per_row = DHEAD / 16;
              const int row_pair =
                  (n >> 1) & (vectors_per_row - 1);
              const int physical_vector = tile ^ row_pair;
              const int physical_d =
                  physical_vector * 16 + group * 4 + i;
              b[nb][i] = shared_k[n * DHEAD + physical_d];
            } else {
              b[nb][i] = shared_k[n * DHEAD + k0 + i];
            }
          }
        }

#pragma unroll
        for (int p = 0; p < MMA_M; ++p) {
#pragma unroll
          for (int nb = 0; nb < QK_MMA_N; ++nb) {
            I8Mma::FragmentC next;
            I8Mma{}(
                next,
                resident_q[p][tile],
                b[nb],
                qk_acc[p][nb]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              qk_acc[p][nb][i] = next[i];
            }
          }
        }
      }

      const float scale =
          q_scale * K_scale[static_cast<size_t>(head) * kv_blocks +
                            kv_block];

#pragma unroll
      for (int p = 0; p < MMA_M; ++p) {
        float local_max = -INFINITY;

#pragma unroll
        for (int nb = 0; nb < QK_MMA_N; ++nb) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float score =
                static_cast<float>(qk_acc[p][nb][i]) * scale;
            local_max = fmaxf(local_max, score);
          }
        }

        float tile_max = local_max;
        tile_max = fmaxf(
            tile_max,
            __shfl_xor_sync(FULL_WAVE_MASK, tile_max, 16, LANES));
        tile_max = fmaxf(
            tile_max,
            __shfl_xor_sync(FULL_WAVE_MASK, tile_max, 32, LANES));

        const float new_m = fmaxf(running_m[p], tile_max);
        float local_sum = 0.0f;

#pragma unroll
        for (int nb = 0; nb < QK_MMA_N; ++nb) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float score =
                static_cast<float>(qk_acc[p][nb][i]) * scale;
            const float probability = __builtin_elementwise_exp2(score - new_m);
            local_sum += probability;
            p_fragment[p][nb][i] = mctlass::half_t(probability);
          }
        }

        float tile_sum = local_sum;
        tile_sum += __shfl_xor_sync(
            FULL_WAVE_MASK, tile_sum, 16, LANES);
        tile_sum += __shfl_xor_sync(
            FULL_WAVE_MASK, tile_sum, 32, LANES);

        const float alpha = exp2f(running_m[p] - new_m);
        running_m[p] = new_m;
        running_l[p] = running_l[p] * alpha + tile_sum;
        row_alpha[p] = alpha;
      }
    }

    if (kv_block != 0) {
#pragma unroll
    for (int p = 0; p < MMA_M; ++p) {
#pragma unroll
      for (int db = 0; db < PV_N_TILES; ++db) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          numerator_acc[p][db][i] *= row_alpha[p];
        }
      }
    }
  }

#pragma unroll
    for (int db = 0; db < PV_N_TILES; ++db) {
#pragma unroll
      for (int kt = 0; kt < PV_K_TILES; ++kt) {
        F16Mma::FragmentB b;
        const int d_for_b = db * 16 + row;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int key = kt * 16 + group * 4 + i;
          if constexpr (STAGE_V) {
            const int swizzle = ((((key >> 2) & 3) << 1) ^ (key & 3));
            const int physical_d = d_for_b ^ (swizzle << 3);
            b[i] = shared_v[static_cast<size_t>(key) * OUT_D +
                            physical_d];
          } else {
            const int global_key = kv_block * CTA_KV + key;
            b[i] = v_head[static_cast<size_t>(global_key) * OUT_D +
                          d_for_b];
          }
        }

        #pragma unroll
      for (int p = 0; p < MMA_M; ++p) {
        F16Mma::FragmentC next;
        F16Mma{}(next, p_fragment[p][kt], b, numerator_acc[p][db]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          numerator_acc[p][db][i] = next[i];
        }
      }
      }
    }
  }

  #pragma unroll
  for (int p = 0; p < MMA_M; ++p) {
    const int output_m = (wave * MMA_M + p) * 16 + row;
#pragma unroll
    for (int db = 0; db < PV_N_TILES; ++db) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = db * 16 + group * 4 + i;
        const size_t index =
            static_cast<size_t>(q_start + output_m) * OUT_D + d;
        o_head[index] =
            mctlass::half_t(numerator_acc[p][db][i] / running_l[p]);
      }
    }
  }
}


namespace py = pybind11;

static void require_argument(bool condition, const char *message) {
  if (!condition) {
    throw py::value_error(message);
  }
}

static void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

static bool aligned(std::uintptr_t address, std::uintptr_t alignment) {
  return address % alignment == 0;
}

static void launch(
    std::uintptr_t q_address,
    std::uintptr_t k_address,
    std::uintptr_t v_address,
    std::uintptr_t q_scale_address,
    std::uintptr_t k_scale_address,
    std::uintptr_t output_address,
    std::int64_t total_heads,
    std::int64_t q_len,
    std::int64_t kv_len,
    std::uintptr_t stream_address) {
  require_argument(q_address != 0, "Q pointer must be non-zero");
  require_argument(k_address != 0, "K pointer must be non-zero");
  require_argument(v_address != 0, "V pointer must be non-zero");
  require_argument(q_scale_address != 0, "Q scale pointer must be non-zero");
  require_argument(k_scale_address != 0, "K scale pointer must be non-zero");
  require_argument(output_address != 0, "output pointer must be non-zero");

  require_argument(aligned(q_address, 16), "Q must be 16B aligned");
  require_argument(aligned(k_address, 16), "K must be 16B aligned");
  require_argument(aligned(v_address, 16), "V must be 16B aligned");
  require_argument(aligned(q_scale_address, 4), "Q scale must be aligned");
  require_argument(aligned(k_scale_address, 4), "K scale must be aligned");
  require_argument(aligned(output_address, 4), "output must be aligned");

  require_argument(total_heads > 0, "total_heads must be positive");
  require_argument(q_len > 0 && q_len % 128 == 0,
                   "q_len must be a positive multiple of 128");
  require_argument(kv_len > 0 && kv_len % 64 == 0,
                   "kv_len must be a positive multiple of 64");
  require_argument(total_heads <= std::numeric_limits<int>::max(),
                   "total_heads exceeds int range");
  require_argument(q_len <= std::numeric_limits<int>::max(),
                   "q_len exceeds int range");
  require_argument(kv_len <= std::numeric_limits<int>::max(),
                   "kv_len exceeds int range");

  const std::int64_t q_tiles_64 = q_len / CTA_M;
  const std::int64_t task_count = total_heads * q_tiles_64;
  require_argument(
      task_count > 0 && task_count <= std::numeric_limits<int>::max(),
      "launch grid exceeds int range");

  const int q_len_i = static_cast<int>(q_len);
  const int kv_len_i = static_cast<int>(kv_len);
  const int q_tiles = static_cast<int>(q_tiles_64);
  const int q_scale_blocks = q_len_i / 128;
  const int kv_blocks = kv_len_i / CTA_KV;
  const int tasks = static_cast<int>(task_count);

  const auto *q = reinterpret_cast<const int8_t *>(q_address);
  const auto *k = reinterpret_cast<const int8_t *>(k_address);
  const auto *v =
      reinterpret_cast<const mctlass::half_t *>(v_address);
  const auto *q_scale =
      reinterpret_cast<const float *>(q_scale_address);
  const auto *k_scale =
      reinterpret_cast<const float *>(k_scale_address);
  auto *output = reinterpret_cast<mctlass::half_t *>(output_address);
  auto stream = reinterpret_cast<cudaStream_t>(stream_address);

  qk_softmax_pv_dynamic_kv_kernel<true, true>
      <<<tasks, THREADS, V_BLOCK_BYTES, stream>>>(
          q,
          k,
          v,
          q_scale,
          k_scale,
          output,
          q_len_i,
          kv_len_i,
          q_tiles,
          q_scale_blocks,
          kv_blocks);

  check_cuda(cudaPeekAtLastError(), "native SageAttention launch");
}

static py::dict kernel_resources() {
  cudaFuncAttributes attributes{};
  check_cuda(
      cudaFuncGetAttributes(
          &attributes,
          qk_softmax_pv_dynamic_kv_kernel<true, true>),
      "cudaFuncGetAttributes");

  int active_blocks = 0;
  check_cuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          qk_softmax_pv_dynamic_kv_kernel<true, true>,
          THREADS,
          V_BLOCK_BYTES),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

  py::dict result;
  result["static_shared_bytes"] = attributes.sharedSizeBytes;
  result["dynamic_shared_bytes"] = V_BLOCK_BYTES;
  result["total_shared_bytes"] =
      attributes.sharedSizeBytes + V_BLOCK_BYTES;
  result["local_bytes"] = attributes.localSizeBytes;
  result["registers"] = attributes.numRegs;
  result["max_threads"] = attributes.maxThreadsPerBlock;
  result["active_blocks"] = active_blocks;
  return result;
}

static py::dict build_info() {
  py::dict result;
  result["implementation"] = "clean_room_mctlass_cta128_mma_m2_resident_q_fast_probability_exp2_k_xor_shared_v_direct_fp16";
  result["source_sha256"] =
      "aa5fe84e4252aa7dbb85ff02db8755453e76ae7887d0304eff526c65f965fd8d";
  result["kernel_sha256"] =
      "46c0a477c64cd2fbf711be6160e83b7009d81e9610186cdc7f6497a3a412b746";
  result["architecture"] = "xcore1000";
  result["layout"] = "packed_head_major";
  result["q_dtype"] = "int8";
  result["k_dtype"] = "int8";
  result["v_dtype"] = "float16";
  result["scale_dtype"] = "float32";
  result["output_dtype"] = "float16";
  result["q_storage"] = "resident_register_fragments";
  result["math_schedule"] = "probability_only_builtin_elementwise_exp2_alpha_precise_exp2";
  result["shared_q_bytes"] = 0;
  result["head_dim"] = DHEAD;
  result["q_block"] = 128;
  result["kv_block"] = CTA_KV;
  result["causal"] = false;
  result["mask"] = false;
  result["lse"] = false;
  return result;
}

PYBIND11_MODULE(_sage_native_attention_cta128m2, module) {
  module.def("launch", &launch);
  module.def("kernel_resources", &kernel_resources);
  module.def("build_info", &build_info);
}
