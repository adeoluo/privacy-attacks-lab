//! PyO3 bindings, built by maturin into the `privacy_attacks_lab.pal_rs` module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::rdp;

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

#[pymodule]
fn pal_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_rdp, m)?)?;
    m.add_function(wrap_pyfunction!(rdp_to_epsilon, m)?)?;
    m.add_function(wrap_pyfunction!(epsilon, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_sigma, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
