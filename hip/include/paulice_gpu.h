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

// C ABI for the HIP gamma backend.
//
// A batch is a set of independent candidate checks. The device propagates each
// candidate's cumulants, counts the generators that contribute to gamma, and
// returns one score per candidate.

#ifndef PAULICE_GPU_H
#define PAULICE_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Gate opcodes, matching `rustiq_core::structures::CliffordGate`.
#define PAULICE_GATE_CNOT 0
#define PAULICE_GATE_CZ 1
#define PAULICE_GATE_H 2
#define PAULICE_GATE_S 3
#define PAULICE_GATE_SD 4
#define PAULICE_GATE_SQRTX 5
#define PAULICE_GATE_SQRTXD 6

/// Noise-model block kinds the device path understands. Every generator inside
/// one block carries the same rate, which is what lets the device reproduce the
/// host's summation order exactly -- see `paulice_gpu_gamma_batch`.
#define PAULICE_NOISE_DEPOLARIZING 0
#define PAULICE_NOISE_READOUT 1

typedef struct PauliceNoiseBlock {
    int32_t kind;
    double rate;
} PauliceNoiseBlock;

/// One batch of candidates. Every pointer is caller-owned host memory, read
/// only for the duration of the call.
///
/// Per-candidate arrays are concatenated and sliced by the `*_offset` arrays,
/// which hold `n_candidates + 1` entries.
typedef struct PauliceGammaBatch {
    int32_t n_candidates;
    int32_t nqubits;

    /// Gates of each candidate's checked circuit, packed by `paulice_pack_gate`.
    const uint32_t *gates;
    const int32_t *gate_offset;
    /// Number of 2-qubit gates in each candidate, i.e. of recorded wire pairs.
    const int32_t *n_two_qubit;

    /// Z-support bitmasks of the rows to propagate, `row_words` words per row.
    /// Post-selected rows are the checks; logical rows are the measured qubits.
    const uint64_t *post_seed;
    const int32_t *post_row_offset;
    const uint64_t *logical_seed;
    const int32_t *logical_row_offset;
    /// Words per row seed mask, i.e. ceil(nqubits / 64).
    int32_t row_words;

    /// Column of each qubit's readout wire, or -1 when that wire is not
    /// recorded (its qubit's last gate is single-qubit). `nqubits` per candidate.
    const int32_t *readout_column;

    /// Noise-model blocks in emission order, shared by every candidate.
    const PauliceNoiseBlock *blocks;
    int32_t n_blocks;

    /// Output: one accumulated rate per candidate. The caller applies `exp`,
    /// so that the transcendental comes from one library rather than two --
    /// double addition is pinned down by IEEE-754, `exp` is not.
    double *out_rates;
} PauliceGammaBatch;

/// Packs a gate into the representation `gates` expects.
static inline uint32_t paulice_pack_gate(uint32_t opcode, uint32_t q0, uint32_t q1) {
    return (opcode << 24) | (q0 << 12) | q1;
}

/// Number of HIP devices, or a negative error code.
int paulice_gpu_device_count(void);

/// Opaque per-device context holding the stream and the scratch buffers.
typedef struct PauliceGpuCtx PauliceGpuCtx;

PauliceGpuCtx *paulice_gpu_create(int device_id);
void paulice_gpu_destroy(PauliceGpuCtx *ctx);

/// Scores a batch. Returns 0 on success and non-zero on any HIP failure, in
/// which case `out_rates` is untouched and the caller falls back to the CPU.
int paulice_gpu_gamma_batch(PauliceGpuCtx *ctx, const PauliceGammaBatch *batch);

#ifdef __cplusplus
}
#endif

#endif // PAULICE_GPU_H
