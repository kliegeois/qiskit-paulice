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

fn main() {
    #[cfg(feature = "gpu")]
    build_hip();
}

/// Compiles the HIP kernels in `hip/` with `hipcc` and links them against the
/// HIP runtime.
///
/// `ROCM_PATH` (default `/opt/rocm`) locates the toolchain, `HIPCC` overrides
/// the compiler, and `PAULICE_GPU_ARCH` (default `gfx950`) takes a
/// comma-separated list of offload targets for a fat binary.
#[cfg(feature = "gpu")]
fn build_hip() {
    println!("cargo:rerun-if-changed=hip/");
    // Without these, changing the toolchain or target env vars would not
    // reliably retrigger a rebuild, leaving stale ROCm paths/arch in place.
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=PAULICE_GPU_ARCH");

    let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| format!("{rocm_path}/bin/hipcc"));
    let arch = std::env::var("PAULICE_GPU_ARCH").unwrap_or_else(|_| "gfx950".to_string());

    let mut build = cc::Build::new();
    build
        .compiler(&hipcc)
        .file("hip/src/paulice_gpu.cpp")
        .file("hip/src/coverage.hip.cpp")
        .include("hip/include")
        // rocPRIM is header-only, so this is the only thing it needs.
        .include(format!("{rocm_path}/include"))
        .cpp(true)
        .flag("-O3")
        .flag("-std=c++17");
    // Trim and drop empties so a list like "gfx950, gfx90a" does not turn into
    // an invalid "--offload-arch= gfx90a" flag.
    for arch in arch.split(',').map(str::trim).filter(|a| !a.is_empty()) {
        build.flag(format!("--offload-arch={arch}"));
    }
    // No fast-math anywhere in here: the device reproduces the host's
    // floating-point accumulation exactly, which reassociation would break.
    build.compile("paulice_gpu");

    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-search=native={rocm_path}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rocm_path}/lib");
}
