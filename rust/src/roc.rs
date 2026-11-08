//! ROC analysis of membership inference scores and its bootstrap.
//!
//! Higher scores mean "member". The curve is swept over distinct thresholds in
//! decreasing order, so tied scores move together (as in scikit-learn).
//! `tpr_at_fpr` is the largest TPR among points with FPR at most the target, the
//! convention of Carlini et al. (2022).

/// Indices that sort `scores` in decreasing order (NaN last).
pub fn argsort_desc(scores: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        let (x, y) = (scores[a], scores[b]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => y.partial_cmp(&x).unwrap(),
        }
    });
    idx
}

/// Tie test that treats NaNs as equal to each other (and 0.0 as equal to -0.0).
fn same(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// `(fpr, tpr, thresholds)` with a leading `(0, 0, +inf)` point; weights count each example.
pub fn roc_curve_sorted(
    scores: &[f64],
    labels: &[bool],
    weights: Option<&[f64]>,
    order: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut pos, mut neg) = (0.0, 0.0);
    for (i, &l) in labels.iter().enumerate() {
        let w = weights.map_or(1.0, |w| w[i]);
        if l {
            pos += w;
        } else {
            neg += w;
        }
    }
    let mut fpr = vec![0.0];
    let mut tpr = vec![0.0];
    let mut thr = vec![f64::INFINITY];
    let (mut tp, mut fp) = (0.0, 0.0);
    for (k, &i) in order.iter().enumerate() {
        let w = weights.map_or(1.0, |w| w[i]);
        if labels[i] {
            tp += w;
        } else {
            fp += w;
        }
        let last = k + 1 == order.len();
        if last || !same(scores[order[k + 1]], scores[i]) {
            fpr.push(if neg > 0.0 { fp / neg } else { f64::NAN });
            tpr.push(if pos > 0.0 { tp / pos } else { f64::NAN });
            thr.push(scores[i]);
        }
    }
    (fpr, tpr, thr)
}

/// Summary statistics of one curve.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub auc: f64,
    /// Best balanced accuracy over thresholds: `max (1 - (fpr + 1 - tpr) / 2)`.
    pub accuracy: f64,
    pub tpr_at: Vec<f64>,
}

pub fn summarize_curve(fpr: &[f64], tpr: &[f64], targets: &[f64]) -> Summary {
    let mut auc = 0.0;
    let mut acc: f64 = 0.0;
    for k in 0..fpr.len() {
        if k > 0 {
            auc += (fpr[k] - fpr[k - 1]) * (tpr[k] + tpr[k - 1]) * 0.5;
        }
        acc = acc.max(1.0 - (fpr[k] + 1.0 - tpr[k]) * 0.5);
    }
    let tpr_at = targets
        .iter()
        .map(|&t| {
            fpr.iter()
                .zip(tpr)
                .filter(|&(&f, _)| f <= t)
                .fold(0.0f64, |best, (_, &v)| best.max(v))
        })
        .collect();
    Summary {
        auc,
        accuracy: acc,
        tpr_at,
    }
}

pub fn summarize(scores: &[f64], labels: &[bool], targets: &[f64]) -> Summary {
    let order = argsort_desc(scores);
    let (f, t, _) = roc_curve_sorted(scores, labels, None, &order);
    summarize_curve(&f, &t, targets)
}

/// SplitMix64: a small, fast, seedable generator (Steele, Lea and Flood 2014).
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, n)` by multiply-shift (bias below 2^-32 for n < 2^32).
    pub fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
    }
}

/// Bootstrap replicates of the summary: resample examples with replacement `reps` times.
///
/// The scores are sorted once; each replicate draws multinomial counts in O(n) and
/// re-sweeps the fixed order with those counts as weights, which is equivalent to
/// sorting the resampled set (tied copies share a threshold either way).
pub fn bootstrap(
    scores: &[f64],
    labels: &[bool],
    targets: &[f64],
    reps: usize,
    seed: u64,
) -> Vec<Summary> {
    let n = scores.len();
    let order = argsort_desc(scores);
    let mut rng = SplitMix64::new(seed);
    let mut weights = vec![0.0; n];
    let mut out = Vec::with_capacity(reps);
    for _ in 0..reps {
        weights.iter_mut().for_each(|w| *w = 0.0);
        for _ in 0..n {
            weights[rng.below(n)] += 1.0;
        }
        let (f, t, _) = roc_curve_sorted(scores, labels, Some(&weights), &order);
        out.push(summarize_curve(&f, &t, targets));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_and_reversed() {
        let s = [0.9, 0.8, 0.2, 0.1];
        let l = [true, true, false, false];
        let r = summarize(&s, &l, &[0.0]);
        assert_eq!(r.auc, 1.0);
        assert_eq!(r.accuracy, 1.0);
        assert_eq!(r.tpr_at, vec![1.0]);
        let rev = [false, false, true, true];
        assert_eq!(summarize(&s, &rev, &[0.0]).auc, 0.0);
    }

    #[test]
    fn ties_move_together() {
        let s = [0.5, 0.5, 0.5, 0.5];
        let l = [true, false, true, false];
        let order = argsort_desc(&s);
        let (f, t, thr) = roc_curve_sorted(&s, &l, None, &order);
        assert_eq!(f, vec![0.0, 1.0]);
        assert_eq!(t, vec![0.0, 1.0]);
        assert_eq!(thr.len(), 2);
        assert_eq!(summarize(&s, &l, &[0.5]).auc, 0.5);
    }

    #[test]
    fn unit_weights_match_unweighted() {
        let s = [0.3, 0.1, 0.7, 0.7, 0.2, 0.9];
        let l = [true, false, true, false, false, true];
        let order = argsort_desc(&s);
        let w = vec![1.0; s.len()];
        assert_eq!(
            roc_curve_sorted(&s, &l, None, &order),
            roc_curve_sorted(&s, &l, Some(&w), &order)
        );
    }

    #[test]
    fn bootstrap_is_deterministic_and_centred() {
        let n = 2000;
        let mut rng = SplitMix64::new(1);
        let s: Vec<f64> = (0..n)
            .map(|_| rng.next_u64() as f64 / u64::MAX as f64)
            .collect();
        let l: Vec<bool> = (0..n).map(|i| s[i] + (i % 3) as f64 * 0.2 > 0.6).collect();
        let a = bootstrap(&s, &l, &[0.01], 50, 7);
        let b = bootstrap(&s, &l, &[0.01], 50, 7);
        assert_eq!(a, b);
        let mean: f64 = a.iter().map(|x| x.auc).sum::<f64>() / 50.0;
        assert!((mean - summarize(&s, &l, &[0.01]).auc).abs() < 0.01);
    }

    #[test]
    fn below_stays_in_range() {
        let mut rng = SplitMix64::new(3);
        assert!((0..10_000).all(|_| rng.below(7) < 7));
    }

    #[test]
    fn signed_zeros_tie() {
        let s = [0.0, -0.0, 0.0, -0.0];
        let l = [true, false, false, true];
        let order = argsort_desc(&s);
        assert_eq!(roc_curve_sorted(&s, &l, None, &order).0.len(), 2);
    }

    #[test]
    fn nan_scores_sort_last() {
        let s = [f64::NAN, 0.2, 0.9];
        assert_eq!(argsort_desc(&s), vec![2, 1, 0]);
    }
}
