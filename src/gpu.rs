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

//! Gamma scoring on a ROCm device.
//!
//! Compiled only with `--features gpu`, and used only when `PAULICE_USE_GPU=1`
//! is also set. The device takes only batches it can score bit-identically to
//! the host; for anything else `gamma_scores` returns `None` and the caller
//! scores on the host. The early returns in `device::gamma_scores` spell out
//! what that rules out (non-empty stabilizers, noise models other than uniform
//! depolarizing and readout, more than `MAX_QUBITS` qubits, a device that is
//! not wave64, and any malformed candidate).

use super::noise_model::UNoiseModel;
use super::pauli::Pauli;
use rustiq_core::structures::CliffordCircuit;

/// The state every candidate in a batch shares.
pub struct GammaContext<'a> {
    pub noise_models: &'a [UNoiseModel],
    pub stabilizers: &'a [Pauli],
    pub measured_qubits: &'a [usize],
}

/// What distinguishes one candidate: the circuit with its check already
/// inserted, and the check rows to post-select on.
///
/// Candidates are described rather than pre-built: the device derives each
/// candidate's generators and cumulants itself, so the host never
/// materialises them for the whole batch.
pub struct GammaCandidate<'a> {
    pub circuit: &'a CliffordCircuit,
    pub check_qubits: &'a [usize],
    pub virtual_zs: &'a [Vec<usize>],
}

/// Scores a batch on the device, one score per candidate in input order, or
/// returns `None` if the device is unavailable or declines the batch.
pub fn gamma_scores(ctx: &GammaContext, candidates: &[GammaCandidate]) -> Option<Vec<f64>> {
    if candidates.is_empty() {
        return Some(Vec::new());
    }
    device::gamma_scores(ctx, candidates)
}

/// Whether the device path is switched on. Read once: this sits on the batch
/// path and `std::env::var` allocates.
pub fn gpu_enabled_by_env() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PAULICE_USE_GPU")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    })
}

pub fn device_count() -> i32 {
    unsafe { device::ffi::paulice_gpu_device_count() }
}

/// CPU/GPU parity. Only compiled with `--features gpu`, and skipped at runtime
/// when no usable device is present (`device_count() <= 0`, or
/// `device::gamma_scores` declining the batch), so it is inert on machines
/// without a GPU and only fails when a device genuinely disagrees with the CPU
/// reference. This is the guard against the propagate/count/reduce kernels or
/// their hardcoded generator enumeration silently drifting from the host walk.
#[cfg(test)]
mod device_parity_tests {
    use super::*;
    use crate::noise_model::{Readout, UNoiseModel, UniformDepolarizing};
    use rustiq_core::structures::{CliffordCircuit, CliffordGate};

    fn models() -> Vec<UNoiseModel> {
        vec![
            UNoiseModel::UniformDepolarizing(UniformDepolarizing::new(0.01)),
            UNoiseModel::Readout(Readout::new(0.02)),
        ]
    }

    /// The host score for each candidate, computed the way
    /// `CheckEvaluator::evaluate` does under `Metric::Gamma`.
    fn host_gamma_scores(ctx: &GammaContext, candidates: &[GammaCandidate]) -> Vec<f64> {
        candidates
            .iter()
            .map(|candidate| {
                let mut coverage =
                    crate::coverage::Coverage::new(candidate.circuit, ctx.noise_models);
                coverage.set_check_cumulants(candidate.check_qubits, candidate.virtual_zs);
                coverage.set_logical_cumulants(ctx.stabilizers, ctx.measured_qubits);
                coverage.gamma_apx()
            })
            .collect()
    }

    /// A small deterministic pseudo-random circuit, so the test pulls in no rng
    /// dependency and reproduces the same batch on every run.
    fn sample_circuit(nqbits: usize, ngates: usize, seed: u64) -> CliffordCircuit {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let other = |q: usize, r: u64, nqbits: usize| {
            let q2 = (r as usize) % nqbits;
            if q2 == q { (q + 1) % nqbits } else { q2 }
        };
        let mut c = CliffordCircuit::new(nqbits);
        for _ in 0..ngates {
            let q = (next() as usize) % nqbits;
            match next() % 6 {
                0 => c.gates.push(CliffordGate::CZ(q, other(q, next(), nqbits))),
                1 => c
                    .gates
                    .push(CliffordGate::CNOT(q, other(q, next(), nqbits))),
                2 => c.gates.push(CliffordGate::H(q)),
                3 => c.gates.push(CliffordGate::S(q)),
                4 => c.gates.push(CliffordGate::SqrtX(q)),
                _ => c.gates.push(CliffordGate::Sd(q)),
            }
        }
        c
    }

    /// Scores one batch of random circuits both ways and asserts equality.
    /// `nmeasured` sets how many logical (and, minus one, post-selected) rows
    /// each candidate carries, which is what packs into a propagation
    /// wavefront -- taking it past 32 and up to 64 is what exercises the full
    /// 64-lane ballot rather than just its first few bits.
    fn assert_parity(nqubits: usize, ngates: usize, nmeasured: usize, salt: u64) -> bool {
        let models = models();
        let measured: Vec<usize> = (0..nmeasured.min(nqubits)).collect();
        let ctx = GammaContext {
            noise_models: &models,
            stabilizers: &[],
            measured_qubits: &measured,
        };

        let circuits: Vec<CliffordCircuit> = (0..8)
            .map(|s| sample_circuit(nqubits, ngates, 0x9E37_79B9_7F4A_7C15 ^ salt ^ s))
            .collect();
        // Several check rows so the post-selected table also spans many rows.
        let check_qubits: Vec<usize> = (0..measured.len().saturating_sub(1).max(1)).collect();
        let virtual_zs: Vec<Vec<Vec<usize>>> = circuits
            .iter()
            .map(|_| check_qubits.iter().map(|_| Vec::new()).collect())
            .collect();
        let candidates: Vec<GammaCandidate> = circuits
            .iter()
            .zip(virtual_zs.iter())
            .map(|(circuit, vzs)| GammaCandidate {
                circuit,
                check_qubits: &check_qubits,
                virtual_zs: vzs,
            })
            .collect();

        let cpu = host_gamma_scores(&ctx, &candidates);
        let gpu = match device::gamma_scores(&ctx, &candidates) {
            Some(scores) => scores,
            // Device present but declined the batch (or the launch failed); the
            // CPU path already covers this, so there is nothing to compare.
            None => return false,
        };

        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "nqubits={nqubits} nmeasured={nmeasured} candidate {i}: cpu {a} vs gpu {b}"
            );
        }
        true
    }

    #[test]
    fn device_matches_cpu_bit_for_bit() {
        if device_count() <= 0 {
            return;
        }
        // Sweep widths and row counts to cover the shapes the kernels
        // specialise on:
        //
        //   * width picks the propagate specialisation -- `QW` is 1 up to 64
        //     qubits, then 2, 3 and 4 at the 128, 192 and 256 boundaries, so
        //     each register-blocking variant gets exercised.
        //   * row count drives the wavefront packing. Past 32 rows a wave32
        //     device would diverge, and past 64 the columns need more than one
        //     word, which brings in the multi-row-block path.
        //
        // A batch the device declines (e.g. a non-wave64 GPU, which
        // `paulice_gpu_create` refuses) just skips -- the CPU path already
        // covers correctness there; this test only asserts that when the device
        // *does* run, it agrees with the CPU bit for bit.
        let mut ran = false;
        for (nqubits, ngates, nmeasured) in [
            (4, 24, 3),
            (16, 60, 16),
            (40, 120, 40),
            (64, 200, 64),
            (128, 300, 128),
            (192, 360, 192),
            (256, 420, 256),
        ] {
            ran |= assert_parity(nqubits, ngates, nmeasured, nqubits as u64);
        }
        // Passing does not by itself mean the device was exercised, so say so
        // rather than letting a green test imply coverage it does not have.
        if !ran {
            eprintln!("note: the device declined every batch; CPU/GPU parity was NOT checked here");
        }
    }
}

/// HIP backend. Builds the flat batch description the kernels consume, runs
/// them, and turns the returned rate sums into scores.
mod device {
    use super::{GammaCandidate, GammaContext};
    use crate::noise_model::UNoiseModel;
    use rustiq_core::structures::{CliffordCircuit, CliffordGate};
    use std::sync::{Mutex, OnceLock};

    /// Matches `MAX_QWORDS * 64` in the kernels.
    const MAX_QUBITS: usize = 256;

    const GATE_CNOT: u32 = 0;
    const GATE_CZ: u32 = 1;
    const GATE_H: u32 = 2;
    const GATE_S: u32 = 3;
    const GATE_SD: u32 = 4;
    const GATE_SQRTX: u32 = 5;
    const GATE_SQRTXD: u32 = 6;

    const NOISE_DEPOLARIZING: i32 = 0;
    const NOISE_READOUT: i32 = 1;

    pub mod ffi {
        use std::os::raw::{c_int, c_void};

        #[repr(C)]
        pub struct NoiseBlock {
            pub kind: i32,
            pub rate: f64,
        }

        #[repr(C)]
        pub struct GammaBatch {
            pub n_candidates: i32,
            pub nqubits: i32,
            pub gates: *const u32,
            pub gate_offset: *const i32,
            pub n_two_qubit: *const i32,
            pub post_seed: *const u64,
            pub post_row_offset: *const i32,
            pub logical_seed: *const u64,
            pub logical_row_offset: *const i32,
            pub row_words: i32,
            pub readout_column: *const i32,
            pub blocks: *const NoiseBlock,
            pub n_blocks: i32,
            pub out_rates: *mut f64,
        }

        #[link(name = "paulice_gpu", kind = "static")]
        unsafe extern "C" {
            pub fn paulice_gpu_device_count() -> c_int;
            pub fn paulice_gpu_create(device_id: c_int) -> *mut c_void;
            pub fn paulice_gpu_destroy(ctx: *mut c_void);
            pub fn paulice_gpu_gamma_batch(ctx: *mut c_void, batch: *const GammaBatch) -> c_int;
        }
    }

    /// The device context is a plain handle; HIP owns the thread safety, and
    /// access is serialised by the mutex around the session.
    struct Session(*mut std::os::raw::c_void);
    unsafe impl Send for Session {}

    impl Drop for Session {
        fn drop(&mut self) {
            unsafe { ffi::paulice_gpu_destroy(self.0) };
        }
    }

    fn session() -> Option<&'static Mutex<Session>> {
        static SESSION: OnceLock<Option<Mutex<Session>>> = OnceLock::new();
        SESSION
            .get_or_init(|| {
                if super::device_count() <= 0 {
                    return None;
                }
                let device = std::env::var("PAULICE_GPU_DEVICE")
                    .ok()
                    .and_then(|v| v.parse::<i32>().ok())
                    .unwrap_or(0);
                let ctx = unsafe { ffi::paulice_gpu_create(device) };
                if ctx.is_null() {
                    None
                } else {
                    Some(Mutex::new(Session(ctx)))
                }
            })
            .as_ref()
    }

    fn opcode(gate: &CliffordGate) -> (u32, u32, u32) {
        match gate {
            CliffordGate::CNOT(i, j) => (GATE_CNOT, *i as u32, *j as u32),
            CliffordGate::CZ(i, j) => (GATE_CZ, *i as u32, *j as u32),
            CliffordGate::H(i) => (GATE_H, *i as u32, 0),
            CliffordGate::S(i) => (GATE_S, *i as u32, 0),
            CliffordGate::Sd(i) => (GATE_SD, *i as u32, 0),
            CliffordGate::SqrtX(i) => (GATE_SQRTX, *i as u32, 0),
            CliffordGate::SqrtXd(i) => (GATE_SQRTXD, *i as u32, 0),
        }
    }

    /// The noise blocks, if every model is one the device knows how to
    /// enumerate. `None` means "score this batch on the CPU".
    ///
    /// Only models whose generators all share a single rate qualify: that is
    /// what lets the device collapse the gamma sum to a count and still
    /// reproduce the host's summation order bit for bit.
    fn blocks(models: &[UNoiseModel]) -> Option<Vec<ffi::NoiseBlock>> {
        if models.is_empty() {
            return None;
        }
        models
            .iter()
            .map(|model| match model {
                UNoiseModel::UniformDepolarizing(m) => Some(ffi::NoiseBlock {
                    kind: NOISE_DEPOLARIZING,
                    rate: m.rate(),
                }),
                UNoiseModel::Readout(m) => Some(ffi::NoiseBlock {
                    kind: NOISE_READOUT,
                    rate: m.rate(),
                }),
                _ => None,
            })
            .collect()
    }

    /// Column of each qubit's readout wire, or -1 when the wire is not recorded.
    ///
    /// Mirrors the host walk's column numbering: the backward walk visits
    /// 2-qubit gates last-first, so the gate with ascending rank `r` owns
    /// columns `2 * (n2q - 1 - r)` and the next one, and the input wires follow
    /// all of them.
    fn readout_columns(circuit: &CliffordCircuit, n2q: usize, out: &mut Vec<i32>) {
        let mut rank_of_gate = vec![-1i32; circuit.gates.len()];
        let mut rank = 0i32;
        for (index, gate) in circuit.gates.iter().enumerate() {
            if gate.arity() == 2 {
                rank_of_gate[index] = rank;
                rank += 1;
            }
        }
        for wire in crate::utils::last_wires(circuit).iter() {
            out.push(match wire {
                crate::wire::Wire::Input(q) => (2 * n2q + q) as i32,
                crate::wire::Wire::GateWire(gate_index, slot) => {
                    let rank = rank_of_gate[*gate_index];
                    if rank < 0 {
                        // A single-qubit gate's wire is never snapshotted.
                        -1
                    } else {
                        2 * (n2q as i32 - 1 - rank) + *slot as i32
                    }
                }
            });
        }
    }

    fn set_bit(mask: &mut [u64], bit: usize) {
        mask[bit / 64] |= 1u64 << (bit % 64);
    }

    pub fn gamma_scores(ctx: &GammaContext, candidates: &[GammaCandidate]) -> Option<Vec<f64>> {
        // Forward-propagated stabilizer rows would need a second walk direction
        // the kernels do not implement; the measured-qubit mode is what the
        // search actually uses.
        if !ctx.stabilizers.is_empty() || ctx.measured_qubits.is_empty() {
            return None;
        }
        let blocks = blocks(ctx.noise_models)?;
        let nqubits = candidates[0].circuit.nqbits;
        if nqubits == 0 || nqubits > MAX_QUBITS {
            return None;
        }
        if candidates.iter().any(|c| c.circuit.nqbits != nqubits) {
            return None;
        }

        // The device path runs ahead of the CPU fallback, so a malformed batch
        // has to bail out to `None` here rather than silently truncating a
        // mismatched zip or panicking in `set_bit` on an out-of-range index.
        if ctx.measured_qubits.iter().any(|&q| q >= nqubits) {
            return None;
        }
        for candidate in candidates {
            if candidate.check_qubits.len() != candidate.virtual_zs.len() {
                return None;
            }
            if candidate.check_qubits.iter().any(|&q| q >= nqubits)
                || candidate.virtual_zs.iter().flatten().any(|&q| q >= nqubits)
            {
                return None;
            }
        }

        let row_words = nqubits.div_ceil(64);
        let total_gates: usize = candidates.iter().map(|c| c.circuit.gates.len()).sum();
        let mut gates: Vec<u32> = Vec::with_capacity(total_gates);
        let mut gate_offset: Vec<i32> = vec![0];
        let mut n_two_qubit: Vec<i32> = Vec::with_capacity(candidates.len());
        let mut post_seed: Vec<u64> = Vec::new();
        let mut post_row_offset: Vec<i32> = vec![0];
        let mut logical_seed: Vec<u64> = Vec::new();
        let mut logical_row_offset: Vec<i32> = vec![0];
        let mut readout_column: Vec<i32> = Vec::with_capacity(candidates.len() * nqubits);

        for candidate in candidates {
            let circuit = candidate.circuit;
            for gate in &circuit.gates {
                let (op, q0, q1) = opcode(gate);
                gates.push((op << 24) | (q0 << 12) | q1);
            }
            gate_offset.push(gates.len() as i32);
            let n2q = circuit.gates.iter().filter(|g| g.arity() == 2).count();
            n_two_qubit.push(n2q as i32);

            // Post-selected rows: Z on the check qubit and on each virtual Z.
            for (check, vzs) in candidate.check_qubits.iter().zip(candidate.virtual_zs) {
                let row = post_seed.len();
                post_seed.resize(row + row_words, 0);
                set_bit(&mut post_seed[row..], *check);
                for qubit in vzs {
                    set_bit(&mut post_seed[row..], *qubit);
                }
            }
            post_row_offset.push((post_seed.len() / row_words) as i32);

            // Logical rows: Z on each measured qubit.
            for qubit in ctx.measured_qubits {
                let row = logical_seed.len();
                logical_seed.resize(row + row_words, 0);
                set_bit(&mut logical_seed[row..], *qubit);
            }
            logical_row_offset.push((logical_seed.len() / row_words) as i32);

            readout_columns(circuit, n2q, &mut readout_column);
        }

        let mut rates = vec![0.0f64; candidates.len()];
        let batch = ffi::GammaBatch {
            n_candidates: candidates.len() as i32,
            nqubits: nqubits as i32,
            gates: gates.as_ptr(),
            gate_offset: gate_offset.as_ptr(),
            n_two_qubit: n_two_qubit.as_ptr(),
            post_seed: post_seed.as_ptr(),
            post_row_offset: post_row_offset.as_ptr(),
            logical_seed: logical_seed.as_ptr(),
            logical_row_offset: logical_row_offset.as_ptr(),
            row_words: row_words as i32,
            readout_column: readout_column.as_ptr(),
            blocks: blocks.as_ptr(),
            n_blocks: blocks.len() as i32,
            out_rates: rates.as_mut_ptr(),
        };

        let session = session()?;
        let guard = session.lock().ok()?;
        let status = unsafe { ffi::paulice_gpu_gamma_batch(guard.0, &batch) };
        drop(guard);
        if status != 0 {
            return None;
        }
        // `exp` is applied here, not on the device: the summed rate is
        // bit-identical either way, but two libm implementations need not agree
        // on the last ulp of the transcendental.
        Some(rates.into_iter().map(|acc| (2.0 * acc).exp()).collect())
    }
}
