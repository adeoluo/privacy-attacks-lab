//! PyO3 bindings, built by maturin into the `privacy_attacks_lab.pal_rs` module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{rdp, roc};

fn check(q: f64, sigma: f64, orders: &[f64]) -> PyResult<()> {
    if !(0.0..=1.0).contains(&q) {
        return Err(PyValueError::new_err(format!(
            "sample rate must be in [0, 1], got {q}"
        )));
    }
    if sigma.is_nan() || sigma < 0.0 {
        return Err(PyValueError::new_err(format!(
            "noise multiplier must be non-negative, got {sigma}"
        )));
    }
    if orders.iter().any(|&a| a.is_nan() || a <= 1.0) {
        return Err(PyValueError::new_err("RDP orders must be greater than 1"));
    }
    Ok(())
}

/// RDP after `steps` steps at each order.
#[pyfunction]
fn compute_rdp(q: f64, sigma: f64, steps: u64, orders: Vec<f64>) -> PyResult<Vec<f64>> {
    check(q, sigma, &orders)?;
    Ok(rdp::compute_rdp(q, sigma, steps, &orders))
}

/// `(epsilon, best order)` from RDP values.
#[pyfunction]
fn rdp_to_epsilon(orders: Vec<f64>, rdp_values: Vec<f64>, delta: f64) -> PyResult<(f64, f64)> {
    if orders.len() != rdp_values.len() {
        return Err(PyValueError::new_err(
            "orders and rdp must have the same length",
        ));
    }
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PyValueError::new_err("delta must be in (0, 1)"));
    }
    Ok(rdp::rdp_to_epsilon(&orders, &rdp_values, delta))
}

/// `(epsilon, best order)` after `steps` steps.
#[pyfunction]
fn epsilon(q: f64, sigma: f64, steps: u64, delta: f64, orders: Vec<f64>) -> PyResult<(f64, f64)> {
    check(q, sigma, &orders)?;
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PyValueError::new_err("delta must be in (0, 1)"));
    }
    Ok(rdp::epsilon(q, sigma, steps, delta, &orders))
}

/// Smallest noise multiplier meeting `target_epsilon`, or None.
#[pyfunction]
fn calibrate_sigma(
    target_epsilon: f64,
    delta: f64,
    q: f64,
    steps: u64,
    orders: Vec<f64>,
    rtol: f64,
) -> PyResult<Option<f64>> {
    check(q, 1.0, &orders)?;
    if target_epsilon.is_nan() || target_epsilon <= 0.0 || rtol.is_nan() || rtol <= 0.0 {
        return Err(PyValueError::new_err(
            "target epsilon and rtol must be positive",
        ));
    }
    Ok(rdp::calibrate_sigma(
        target_epsilon,
        delta,
        q,
        steps,
        &orders,
        rtol,
    ))
}

fn f64s(buf: &[u8], what: &str) -> PyResult<Vec<f64>> {
    if !buf.len().is_multiple_of(8) {
        return Err(PyValueError::new_err(format!(
            "{what}: buffer is not float64"
        )));
    }
    Ok(buf
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_ne_bytes(*c))
        .collect())
}

fn scores_labels(scores: &[u8], labels: &[u8]) -> PyResult<(Vec<f64>, Vec<bool>)> {
    let s = f64s(scores, "scores")?;
    if s.len() != labels.len() {
        return Err(PyValueError::new_err(
            "scores and labels must have the same length",
        ));
    }
    Ok((s, labels.iter().map(|&b| b != 0).collect()))
}

/// ROC curve `(fpr, tpr, thresholds)` from float64 scores and uint8 labels (raw bytes).
#[pyfunction]
fn roc_curve(scores: &[u8], labels: &[u8]) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let (s, l) = scores_labels(scores, labels)?;
    let order = roc::argsort_desc(&s);
    Ok(roc::roc_curve_sorted(&s, &l, None, &order))
}

/// `(auc, accuracy, [tpr at each target fpr])`.
#[pyfunction]
fn roc_summary(scores: &[u8], labels: &[u8], targets: Vec<f64>) -> PyResult<(f64, f64, Vec<f64>)> {
    let (s, l) = scores_labels(scores, labels)?;
    let r = roc::summarize(&s, &l, &targets);
    Ok((r.auc, r.accuracy, r.tpr_at))
}

/// Bootstrap replicates: a list of `(auc, accuracy, [tpr at targets])`.
#[pyfunction]
fn roc_bootstrap(
    py: Python<'_>,
    scores: &[u8],
    labels: &[u8],
    targets: Vec<f64>,
    reps: usize,
    seed: u64,
) -> PyResult<Vec<(f64, f64, Vec<f64>)>> {
    let (s, l) = scores_labels(scores, labels)?;
    let out = py.detach(|| roc::bootstrap(&s, &l, &targets, reps, seed));
    Ok(out
        .into_iter()
        .map(|r| (r.auc, r.accuracy, r.tpr_at))
        .collect())
}

#[pymodule]
fn pal_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_rdp, m)?)?;
    m.add_function(wrap_pyfunction!(rdp_to_epsilon, m)?)?;
    m.add_function(wrap_pyfunction!(epsilon, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(roc_curve, m)?)?;
    m.add_function(wrap_pyfunction!(roc_summary, m)?)?;
    m.add_function(wrap_pyfunction!(roc_bootstrap, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
