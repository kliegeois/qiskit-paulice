// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// Host side of the HIP gamma backend: device context, scratch buffers, and the
// upload/launch/download sequence for one batch.

#include "paulice_gpu.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <math.h>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

extern "C" hipError_t paulice_launch_propagate(
    const uint32_t *gates, const int32_t *gate_offset, const int32_t *n_two_qubit,
    const uint64_t *seeds, const int32_t *row_offset, int row_words, int nqubits,
    const int64_t *word_offset, const int32_t *nwords_per_candidate, uint64_t *x_out,
    uint64_t *z_out, int n_candidates, int max_row_blocks, hipStream_t stream);

extern "C" hipError_t paulice_launch_count(
    const int32_t *n_two_qubit, int nqubits, const int32_t *readout_column,
    const PauliceNoiseBlock *blocks, int n_blocks, const uint64_t *post_x, const uint64_t *post_z,
    const int64_t *post_word_offset, const int32_t *post_nwords, const int32_t *post_row_offset,
    const uint64_t *log_x, const uint64_t *log_z, const int64_t *log_word_offset,
    const int32_t *log_nwords, const int32_t *log_row_offset, int32_t *counts_out,
    int n_candidates, hipStream_t stream);

extern "C" hipError_t paulice_launch_reduce(const int32_t *counts, const PauliceNoiseBlock *blocks,
                                            int n_blocks, int n_candidates, double *out_rates,
                                            hipStream_t stream);

namespace {

/// A device allocation that grows but never shrinks, so a steady-state batch
/// loop stops calling hipMalloc entirely.
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer() { release(); }

    void release() {
        if (ptr_ != nullptr) {
            (void)hipFree(ptr_);
            ptr_ = nullptr;
        }
        capacity_ = 0;
    }

    /// Ensures room for `bytes`, returning false if the allocation failed.
    bool ensure(size_t bytes) {
        if (bytes <= capacity_) {
            return true;
        }
        release();
        if (hipMalloc(&ptr_, bytes) != hipSuccess) {
            ptr_ = nullptr;
            capacity_ = 0;
            return false;
        }
        capacity_ = bytes;
        return true;
    }

    template <typename T> T *as() { return static_cast<T *>(ptr_); }
    template <typename T> const T *as() const { return static_cast<const T *>(ptr_); }

  private:
    void *ptr_ = nullptr;
    size_t capacity_ = 0;
};

/// Uploads `count` elements, growing `buffer` first.
template <typename T>
bool upload(DeviceBuffer &buffer, const T *src, size_t count, hipStream_t stream) {
    if (!buffer.ensure(count * sizeof(T))) {
        return false;
    }
    if (count == 0) {
        return true;
    }
    return hipMemcpyAsync(buffer.as<T>(), src, count * sizeof(T), hipMemcpyHostToDevice, stream) ==
           hipSuccess;
}

/// Per-stage timing, printed when PAULICE_GPU_PROFILE is set. Off by default:
/// it forces a synchronise between stages.
bool profiling() {
    static const bool on = [] {
        const char *v = getenv("PAULICE_GPU_PROFILE");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

class StageTimer {
  public:
    explicit StageTimer(hipStream_t stream) : stream_(stream) {
        if (!profiling()) {
            return;
        }
        (void)hipEventCreate(&start_);
        (void)hipEventCreate(&stop_);
        (void)hipEventRecord(start_, stream_);
    }

    ~StageTimer() {
        if (!profiling()) {
            return;
        }
        (void)hipEventDestroy(start_);
        (void)hipEventDestroy(stop_);
    }

    void report(const char *name) {
        if (!profiling()) {
            return;
        }
        (void)hipEventRecord(stop_, stream_);
        (void)hipEventSynchronize(stop_);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, start_, stop_);
        fprintf(stderr, "[paulice-gpu] %-12s %8.3f ms\n", name, ms);
        (void)hipEventRecord(start_, stream_);
    }

  private:
    hipStream_t stream_;
    hipEvent_t start_ = nullptr;
    hipEvent_t stop_ = nullptr;
};

} // namespace

struct PauliceGpuCtx {
    int device_id = 0;
    hipStream_t stream = nullptr;
    /// The two cumulant tables are independent walks, so the second one runs on
    /// its own stream. Each is only a few hundred wavefronts -- far short of
    /// filling the device -- so overlapping them is close to free parallelism.
    hipStream_t alt_stream = nullptr;
    hipEvent_t fork = nullptr;
    hipEvent_t join = nullptr;

    DeviceBuffer gates;
    DeviceBuffer gate_offset;
    DeviceBuffer n_two_qubit;
    DeviceBuffer post_seed;
    DeviceBuffer post_row_offset;
    DeviceBuffer logical_seed;
    DeviceBuffer logical_row_offset;
    DeviceBuffer readout_column;
    DeviceBuffer blocks;

    DeviceBuffer post_x;
    DeviceBuffer post_z;
    DeviceBuffer post_word_offset;
    DeviceBuffer post_nwords;
    DeviceBuffer log_x;
    DeviceBuffer log_z;
    DeviceBuffer log_word_offset;
    DeviceBuffer log_nwords;
    DeviceBuffer counts;
    DeviceBuffer rates;
};

int paulice_gpu_device_count(void) {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) {
        return -1;
    }
    return count;
}

PauliceGpuCtx *paulice_gpu_create(int device_id) {
    if (hipSetDevice(device_id) != hipSuccess) {
        return nullptr;
    }
    // The kernels are written for a 64-wide wavefront: a ballot packs one
    // cumulant row per lane into a 64-bit column word, and a workgroup is one
    // wavefront. On a 32-wide device (e.g. RDNA) that packing is wrong and the
    // scores come out silently incorrect, so refuse the device here and let the
    // caller fall back to the CPU rather than return garbage.
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id) != hipSuccess || props.warpSize != 64) {
        return nullptr;
    }
    PauliceGpuCtx *ctx = new (std::nothrow) PauliceGpuCtx();
    if (ctx == nullptr) {
        return nullptr;
    }
    ctx->device_id = device_id;
    if (hipStreamCreateWithFlags(&ctx->stream, hipStreamNonBlocking) != hipSuccess ||
        hipStreamCreateWithFlags(&ctx->alt_stream, hipStreamNonBlocking) != hipSuccess ||
        hipEventCreateWithFlags(&ctx->fork, hipEventDisableTiming) != hipSuccess ||
        hipEventCreateWithFlags(&ctx->join, hipEventDisableTiming) != hipSuccess) {
        paulice_gpu_destroy(ctx);
        return nullptr;
    }
    return ctx;
}

void paulice_gpu_destroy(PauliceGpuCtx *ctx) {
    if (ctx == nullptr) {
        return;
    }
    (void)hipSetDevice(ctx->device_id);
    if (ctx->join != nullptr) {
        (void)hipEventDestroy(ctx->join);
    }
    if (ctx->fork != nullptr) {
        (void)hipEventDestroy(ctx->fork);
    }
    if (ctx->alt_stream != nullptr) {
        (void)hipStreamDestroy(ctx->alt_stream);
    }
    if (ctx->stream != nullptr) {
        (void)hipStreamDestroy(ctx->stream);
    }
    delete ctx;
}

int paulice_gpu_gamma_batch(PauliceGpuCtx *ctx, const PauliceGammaBatch *batch) {
    if (ctx == nullptr || batch == nullptr || batch->n_candidates <= 0 || batch->n_blocks <= 0) {
        return -1;
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess) {
        return -1;
    }

    const int n = batch->n_candidates;
    const int nq = batch->nqubits;
    const hipStream_t stream = ctx->stream;
    StageTimer timer(stream);

    // Every candidate records two columns per 2-qubit gate plus one per input
    // wire, and both tables share that column layout since both walk the same
    // circuit. The number of *words* per column differs though -- it follows
    // each table's row count -- so the word offsets are accumulated explicitly
    // rather than derived from a column count and a single stride.
    std::vector<int64_t> post_word_offset(n + 1, 0);
    std::vector<int64_t> log_word_offset(n + 1, 0);
    std::vector<int32_t> post_nwords(n, 1);
    std::vector<int32_t> log_nwords(n, 1);
    int max_post_blocks = 1;
    int max_log_blocks = 1;
    for (int c = 0; c < n; ++c) {
        const int64_t columns = 2 * batch->n_two_qubit[c] + nq;
        const int post_rows = batch->post_row_offset[c + 1] - batch->post_row_offset[c];
        const int log_rows = batch->logical_row_offset[c + 1] - batch->logical_row_offset[c];
        post_nwords[c] = std::max(1, (post_rows + 63) / 64);
        log_nwords[c] = std::max(1, (log_rows + 63) / 64);
        post_word_offset[c + 1] = post_word_offset[c] + columns * post_nwords[c];
        log_word_offset[c + 1] = log_word_offset[c] + columns * log_nwords[c];
        max_post_blocks = std::max(max_post_blocks, post_nwords[c]);
        max_log_blocks = std::max(max_log_blocks, log_nwords[c]);
    }

    const size_t n_gates = static_cast<size_t>(batch->gate_offset[n]);
    const size_t n_post_rows = static_cast<size_t>(batch->post_row_offset[n]);
    const size_t n_log_rows = static_cast<size_t>(batch->logical_row_offset[n]);
    const size_t row_words = static_cast<size_t>(batch->row_words);

    if (!upload(ctx->gates, batch->gates, n_gates, stream) ||
        !upload(ctx->gate_offset, batch->gate_offset, static_cast<size_t>(n) + 1, stream) ||
        !upload(ctx->n_two_qubit, batch->n_two_qubit, static_cast<size_t>(n), stream) ||
        !upload(ctx->post_seed, batch->post_seed, n_post_rows * row_words, stream) ||
        !upload(ctx->post_row_offset, batch->post_row_offset, static_cast<size_t>(n) + 1, stream) ||
        !upload(ctx->logical_seed, batch->logical_seed, n_log_rows * row_words, stream) ||
        !upload(ctx->logical_row_offset, batch->logical_row_offset, static_cast<size_t>(n) + 1,
                stream) ||
        !upload(ctx->readout_column, batch->readout_column, static_cast<size_t>(n) * nq, stream) ||
        !upload(ctx->blocks, batch->blocks, static_cast<size_t>(batch->n_blocks), stream) ||
        !upload(ctx->post_word_offset, post_word_offset.data(), static_cast<size_t>(n) + 1,
                stream) ||
        !upload(ctx->log_word_offset, log_word_offset.data(), static_cast<size_t>(n) + 1, stream) ||
        !upload(ctx->post_nwords, post_nwords.data(), static_cast<size_t>(n), stream) ||
        !upload(ctx->log_nwords, log_nwords.data(), static_cast<size_t>(n), stream)) {
        return -1;
    }
    timer.report("upload");

    // Fan the two propagations out across both streams, then join before the
    // counting pass, which reads from both tables.
    if (hipEventRecord(ctx->fork, stream) != hipSuccess ||
        hipStreamWaitEvent(ctx->alt_stream, ctx->fork, 0) != hipSuccess) {
        return -1;
    }

    const size_t post_words = static_cast<size_t>(post_word_offset[n]);
    const size_t log_words = static_cast<size_t>(log_word_offset[n]);
    if (!ctx->post_x.ensure(post_words * sizeof(uint64_t)) ||
        !ctx->post_z.ensure(post_words * sizeof(uint64_t)) ||
        !ctx->log_x.ensure(log_words * sizeof(uint64_t)) ||
        !ctx->log_z.ensure(log_words * sizeof(uint64_t)) ||
        !ctx->counts.ensure(static_cast<size_t>(n) * batch->n_blocks * sizeof(int32_t)) ||
        !ctx->rates.ensure(static_cast<size_t>(n) * sizeof(double))) {
        return -1;
    }

    if (paulice_launch_propagate(ctx->gates.as<uint32_t>(), ctx->gate_offset.as<int32_t>(),
                                 ctx->n_two_qubit.as<int32_t>(), ctx->post_seed.as<uint64_t>(),
                                 ctx->post_row_offset.as<int32_t>(), batch->row_words, nq,
                                 ctx->post_word_offset.as<int64_t>(),
                                 ctx->post_nwords.as<int32_t>(), ctx->post_x.as<uint64_t>(),
                                 ctx->post_z.as<uint64_t>(), n, max_post_blocks,
                                 stream) != hipSuccess) {
        return -1;
    }
    if (paulice_launch_propagate(ctx->gates.as<uint32_t>(), ctx->gate_offset.as<int32_t>(),
                                 ctx->n_two_qubit.as<int32_t>(), ctx->logical_seed.as<uint64_t>(),
                                 ctx->logical_row_offset.as<int32_t>(), batch->row_words, nq,
                                 ctx->log_word_offset.as<int64_t>(), ctx->log_nwords.as<int32_t>(),
                                 ctx->log_x.as<uint64_t>(), ctx->log_z.as<uint64_t>(), n,
                                 max_log_blocks, ctx->alt_stream) != hipSuccess) {
        return -1;
    }
    if (hipEventRecord(ctx->join, ctx->alt_stream) != hipSuccess ||
        hipStreamWaitEvent(stream, ctx->join, 0) != hipSuccess) {
        return -1;
    }
    timer.report("propagate");
    if (paulice_launch_count(ctx->n_two_qubit.as<int32_t>(), nq, ctx->readout_column.as<int32_t>(),
                             ctx->blocks.as<PauliceNoiseBlock>(), batch->n_blocks,
                             ctx->post_x.as<uint64_t>(), ctx->post_z.as<uint64_t>(),
                             ctx->post_word_offset.as<int64_t>(), ctx->post_nwords.as<int32_t>(),
                             ctx->post_row_offset.as<int32_t>(), ctx->log_x.as<uint64_t>(),
                             ctx->log_z.as<uint64_t>(), ctx->log_word_offset.as<int64_t>(),
                             ctx->log_nwords.as<int32_t>(), ctx->logical_row_offset.as<int32_t>(),
                             ctx->counts.as<int32_t>(), n, stream) != hipSuccess) {
        return -1;
    }
    timer.report("count");
    if (paulice_launch_reduce(ctx->counts.as<int32_t>(), ctx->blocks.as<PauliceNoiseBlock>(),
                              batch->n_blocks, n, ctx->rates.as<double>(), stream) != hipSuccess) {
        return -1;
    }

    if (hipMemcpyAsync(batch->out_rates, ctx->rates.as<double>(),
                       static_cast<size_t>(n) * sizeof(double), hipMemcpyDeviceToHost,
                       stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        return -1;
    }
    timer.report("reduce+dl");
    return 0;
}
