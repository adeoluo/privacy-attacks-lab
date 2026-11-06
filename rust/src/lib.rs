//! Rust core of privacy-attacks-lab.
//!
//! - [`rdp`]: Renyi DP of the Poisson subsampled Gaussian mechanism, conversion to
//!   (epsilon, delta) and noise calibration.
//! - [`roc`]: ROC curves, AUC, TPR at fixed FPR and their bootstrap.
//!
//! The Python package keeps a reference implementation of every function, and the
//! parity tests check that both agree.

pub mod rdp;
pub mod roc;

#[cfg(feature = "python")]
mod python;
