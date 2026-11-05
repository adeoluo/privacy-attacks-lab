//! Renyi DP of the Poisson subsampled Gaussian mechanism, its conversion to
//! (epsilon, delta) and noise calibration.
//!
//! Mirrors `privacy_attacks_lab.accountants.rdp` (the Python reference); the
//! parity tests check both agree to about 1e-12 relative error.

const SERIES_CUTOFF: f64 = -30.0;

/// `log(exp(x) + exp(y))` without overflow.
pub fn log_add(x: f64, y: f64) -> f64 {
    let (a, b) = if x < y { (x, y) } else { (y, x) };
    if a == f64::NEG_INFINITY {
        return b;
    }
    (a - b).exp().ln_1p() + b
}

/// `log(exp(x) - exp(y))` for `x >= y`.
pub fn log_sub(x: f64, y: f64) -> f64 {
    debug_assert!(x >= y, "log_sub needs x >= y");
    if y == f64::NEG_INFINITY {
        return x;
    }
    if x == y {
        return f64::NEG_INFINITY;
    }
    let d = (x - y).exp_m1();
    if d.is_infinite() {
        return x;
    }
    d.ln() + y
}

/// `log(erfc(x))`, accurate where `erfc(x)` underflows (asymptotic series past 26).
pub fn log_erfc(x: f64) -> f64 {
    if x < 26.0 {
        return libm::erfc(x).ln();
    }
    // erfc(x) = exp(-x^2) / (x sqrt(pi)) * sum_n (-1)^n (2n - 1)!! / (2 x^2)^n
    let inv = 1.0 / (2.0 * x * x);
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 1..12 {
        term *= -((2 * n - 1) as f64) * inv;
        sum += term;
    }
    -x * x - x.ln() - 0.5 * std::f64::consts::PI.ln() + sum.ln()
}

fn ln_gamma_signed(x: f64) -> (f64, f64) {
    let (v, s) = libm::lgamma_r(x);
    (v, if s < 0 { -1.0 } else { 1.0 })
}

/// `(log |C(n, k)|, sign)` for real `n` and integer `k >= 0`.
fn log_binom(n: f64, k: u64) -> (f64, f64) {
    let kf = k as f64;
    if n.fract() == 0.0 && n >= 0.0 && kf > n {
        return (f64::NEG_INFINITY, 0.0);
    }
    let (a, sa) = ln_gamma_signed(n + 1.0);
    let (b, _) = ln_gamma_signed(kf + 1.0);
    let (c, sc) = ln_gamma_signed(n - kf + 1.0);
    (a - b - c, sa * sc)
}

/// `log A_alpha` for integer `alpha` (Mironov et al. 2019, Section 3.3).
pub fn log_a_int(q: f64, sigma: f64, alpha: u64) -> f64 {
    let (lq, l1q) = (q.ln(), (-q).ln_1p());
    let inv = 1.0 / (2.0 * sigma * sigma);
    let a = alpha as f64;
    let mut acc = f64::NEG_INFINITY;
    for i in 0..=alpha {
        let fi = i as f64;
        let (lb, _) = log_binom(a, i);
        acc = log_add(acc, lb + fi * lq + (a - fi) * l1q + (fi * fi - fi) * inv);
    }
    acc
}

/// `log A_alpha` for fractional `alpha`: two series, stopped once both terms drop below e^-30.
pub fn log_a_frac(q: f64, sigma: f64, alpha: f64) -> f64 {
    let (mut a0, mut a1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let s2 = sigma * sigma;
    let z0 = s2 * (1.0 / q - 1.0).ln() + 0.5;
    let (lq, l1q) = (q.ln(), (-q).ln_1p());
    let r2s = std::f64::consts::SQRT_2 * sigma;
    let log_half = 0.5f64.ln();
    let mut i: u64 = 0;
    loop {
        let fi = i as f64;
        let (lc, sign) = log_binom(alpha, i);
        let j = alpha - fi;
        let t0 = lc + fi * lq + j * l1q;
        let t1 = lc + j * lq + fi * l1q;
        let e0 = log_half + log_erfc((fi - z0) / r2s);
        let e1 = log_half + log_erfc((z0 - j) / r2s);
        let s0 = t0 + (fi * fi - fi) / (2.0 * s2) + e0;
        let s1 = t1 + (j * j - j) / (2.0 * s2) + e1;
        if sign > 0.0 {
            a0 = log_add(a0, s0);
            a1 = log_add(a1, s1);
        } else {
            a0 = log_sub(a0, s0);
            a1 = log_sub(a1, s1);
        }
        i += 1;
        if s0.max(s1) < SERIES_CUTOFF {
            break;
        }
    }
    log_add(a0, a1)
}

/// RDP at order `alpha` of one step with sampling rate `q` and noise multiplier `sigma`.
pub fn rdp_single(q: f64, sigma: f64, alpha: f64) -> f64 {
    if q == 0.0 {
        return 0.0;
    }
    if sigma == 0.0 || alpha.is_infinite() {
        return f64::INFINITY;
    }
    if q == 1.0 {
        return alpha / (2.0 * sigma * sigma);
    }
    if alpha.fract() == 0.0 {
        log_a_int(q, sigma, alpha as u64) / (alpha - 1.0)
    } else {
        log_a_frac(q, sigma, alpha) / (alpha - 1.0)
    }
}

/// RDP after `steps` compositions at every order.
pub fn compute_rdp(q: f64, sigma: f64, steps: u64, orders: &[f64]) -> Vec<f64> {
    orders
        .iter()
        .map(|&a| rdp_single(q, sigma, a) * steps as f64)
        .collect()
}

/// `(epsilon, best order)` at `delta` (Balle et al. 2020, Theorem 21); NaN orders are skipped.
pub fn rdp_to_epsilon(orders: &[f64], rdp: &[f64], delta: f64) -> (f64, f64) {
    let ld = delta.ln();
    let mut best = (f64::INFINITY, f64::NAN);
    let mut found = false;
    for (&a, &r) in orders.iter().zip(rdp) {
        let e = r - (ld + a.ln()) / (a - 1.0) + ((a - 1.0) / a).ln();
        if e.is_nan() {
            continue;
        }
        if !found || e < best.0 {
            best = (e, a);
            found = true;
        }
    }
    best
}

/// Epsilon at `delta` after `steps` steps.
pub fn epsilon(q: f64, sigma: f64, steps: u64, delta: f64, orders: &[f64]) -> (f64, f64) {
    rdp_to_epsilon(orders, &compute_rdp(q, sigma, steps, orders), delta)
}

/// Smallest noise multiplier (to relative tolerance `rtol`) whose epsilon is at most
/// `target`; `None` if even sigma = 1e6 is not enough.
pub fn calibrate_sigma(
    target: f64,
    delta: f64,
    q: f64,
    steps: u64,
    orders: &[f64],
    rtol: f64,
) -> Option<f64> {
    let eps_at = |s: f64| epsilon(q, s, steps, delta, orders).0;
    let mut hi = 1.0;
    while eps_at(hi) > target {
        hi *= 2.0;
        if hi > 1e6 {
            return None;
        }
    }
    let mut lo = if hi > 1.0 { hi / 2.0 } else { 0.0 };
    while hi - lo > rtol * hi {
        let mid = 0.5 * (lo + hi);
        if eps_at(mid) <= target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_orders() -> Vec<f64> {
        let mut v: Vec<f64> = (1..100).map(|x| 1.0 + x as f64 / 10.0).collect();
        v.extend((12..64).map(|a| a as f64));
        v
    }

    #[test]
    fn log_add_and_sub_invert() {
        let s = log_add(1.0_f64.ln(), 2.0_f64.ln());
        assert!((s - 3.0_f64.ln()).abs() < 1e-15);
        assert!((log_sub(s, 2.0_f64.ln())).abs() < 1e-15);
        assert_eq!(log_sub(1.0, 1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn log_erfc_is_continuous_at_the_switch() {
        let below = libm::erfc(25.999).ln();
        let above = log_erfc(26.0);
        assert!((above - below).abs() < 0.06);
        assert!(
            (log_erfc(30.0) - (-900.0 - 30.0_f64.ln() - 0.5 * std::f64::consts::PI.ln())).abs()
                < 1e-3
        );
    }

    #[test]
    fn full_batch_is_the_gaussian_mechanism() {
        assert!((rdp_single(1.0, 2.0, 3.0) - 3.0 / 8.0).abs() < 1e-15);
        assert_eq!(rdp_single(0.0, 1.0, 3.0), 0.0);
        assert!(rdp_single(0.5, 0.0, 3.0).is_infinite());
    }

    #[test]
    fn rdp_increases_with_order_and_rate() {
        let o = default_orders();
        let r = compute_rdp(0.01, 1.0, 1, &o);
        for w in r.windows(2) {
            assert!(w[1] >= w[0] - 1e-12);
        }
        let more = compute_rdp(0.02, 1.0, 1, &o);
        assert!(more.iter().zip(&r).all(|(a, b)| a >= b));
    }

    #[test]
    fn known_epsilon() {
        // DP-SGD on MNIST in Abadi et al.: q = 256/60000, sigma = 1.1, 60 epochs.
        let o = default_orders();
        let steps = (60.0 * 60000.0 / 256.0) as u64;
        let (e, _) = epsilon(256.0 / 60000.0, 1.1, steps, 1e-5, &o);
        assert!(e > 2.5 && e < 3.0, "eps = {e}");
    }

    #[test]
    fn calibration_meets_the_target() {
        let o = default_orders();
        let s = calibrate_sigma(2.0, 1e-5, 0.01, 1000, &o, 1e-6).unwrap();
        let (e, _) = epsilon(0.01, s, 1000, 1e-5, &o);
        assert!(e <= 2.0 && e > 1.99);
        assert!(calibrate_sigma(1e-9, 1e-5, 1.0, 1_000_000, &o, 1e-3).is_none());
    }
}
