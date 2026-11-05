//! Rust core of privacy-attacks-lab.
//!
//! - [`rdp`]: Renyi DP of the Poisson subsampled Gaussian mechanism, conversion to
//!   (epsilon, delta) and noise calibration.
//!
//! The Python package keeps a reference implementation of every function, and the
//! parity tests check that both agree.

pub mod rdp;

#[cfg(feature = "python")]
mod python;
