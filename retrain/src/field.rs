//! The mean-field (load-aware) routing proof.
//!
//! Paper 2's mean-field claim: a static best-arm policy ignores congestion — if every
//! request routes to the globally-best arm it piles onto one queue, and quality per
//! request is really *population-dependent* (Wardrop routing). A load-aware policy that
//! best-responds to the arm-*load* field should hold realized quality while cutting
//! hot-arm concentration and tail latency.
//!
//! This is testable *exactly* here because the counterfactual corpus carries every
//! arm's outcome AND latency on every prompt ([`Prompt`]), so any policy's realized
//! quality and the load distribution it induces are both computable with no live serving.
//!
//! Model. Each arm's load is its share of the routed population `rho_a = n_a / N`
//! (uniform routing => `rho = 1/|A|`). Congestion inflates an arm's effective latency by
//! the M/M/1 sojourn factor [`hanzo_router::congestion`] `c(rho) = rho/(1-rho)`:
//! `eff_latency = base_latency * (1 + c(rho_a))`. A load-aware pick scores
//! `base_head_score - gamma * c(rho_a)` ([`Heads::best_field`]); the induced `rho`
//! depends on every pick, so we solve the Wardrop fixed point by damped best response.
//!
//! Three policies over the same prompts: (a) **static** head (`gamma = 0`),
//! (b) **load-aware** head (`gamma` swept), (c) **oracle** (per-prompt best quality —
//! the quality ceiling). `best-single` (everyone to the train-best arm) is reported as
//! the pathological-concentration reference.

use std::collections::BTreeMap;

use hanzo_router::heads::{Arm, LoadField};
use hanzo_router::{congestion, Heads, HashFeaturizer, Featurizer, Slo};
use router_learner::{ingest, EvalSample};

use crate::decision::Prompt;
use crate::fit_policy;

/// Per-arm realized signals for one prompt: the counterfactual quality cell plus the
/// congestion inputs (base latency, cost), averaged over the prompt's repeat runs.
#[derive(Debug, Clone, Default)]
pub struct Cell {
    pub quality: f64,
    pub latency_ms: f64,
    pub cost: f64,
}

/// One prompt as the simulation sees it: cached feature vector `x` and the per-arm
/// realized cells. Built once from a [`Prompt`] so the best-response loop is cheap.
pub struct SimPrompt {
    pub x: Vec<f64>,
    pub cells: BTreeMap<String, Cell>,
}

/// Fold a prompt's eval rows into per-arm cells (quality already averaged in
/// `arm_quality`; latency/cost averaged over the rows of each arm).
fn to_cells(p: &Prompt) -> BTreeMap<String, Cell> {
    let mut sum: BTreeMap<String, (f64, f64, u32)> = BTreeMap::new();
    for r in &p.rows {
        let e = sum.entry(r.model.clone()).or_insert((0.0, 0.0, 0));
        e.0 += r.latency_ms;
        e.1 += r.cost;
        e.2 += 1;
    }
    p.arm_quality
        .iter()
        .map(|(a, &q)| {
            let (lat, cost, n) = sum.get(a).copied().unwrap_or((0.0, 0.0, 1));
            let n = n.max(1) as f64;
            (
                a.clone(),
                Cell {
                    quality: q,
                    latency_ms: lat / n,
                    cost: cost / n,
                },
            )
        })
        .collect()
}

/// Build the sim prompts (cached `x` + cells) and the sorted arm list.
pub fn build(prompts: &[Prompt], idx: &[usize]) -> (Vec<SimPrompt>, Vec<String>) {
    let feat = HashFeaturizer::default();
    let sims: Vec<SimPrompt> = idx
        .iter()
        .map(|&i| SimPrompt {
            x: feat.featurize(&prompts[i].request),
            cells: to_cells(&prompts[i]),
        })
        .collect();
    let mut arms: Vec<String> = sims
        .iter()
        .flat_map(|s| s.cells.keys().cloned())
        .collect();
    arms.sort();
    arms.dedup();
    (sims, arms)
}

/// Fit the learned head on `train` prompts exactly as the serve path does (ingest ->
/// profile table -> ridge fit -> mounted `Heads`). `ridge` is the fit regularization
/// (the shipped bundle uses 0.1) — distinct from the mean-field `gamma`.
pub fn fit_head(prompts: &[Prompt], train: &[usize], ridge: f64) -> Heads {
    let feat = HashFeaturizer::default();
    let train_rows: Vec<EvalSample> = train
        .iter()
        .flat_map(|&i| prompts[i].rows.clone())
        .collect();
    let table = ingest(&train_rows);
    let fitted = fit_policy(&train_rows, &table, &feat, ridge);
    let arm_feats: Vec<Arm> = table
        .profiles
        .iter()
        .map(|p| Arm {
            model: p.model.clone(),
            feat: p.features().to_vec(),
        })
        .collect();
    Heads::new(fitted.w, arm_feats)
}

/// The realized metrics of one routing assignment over the population.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub quality: f64,
    /// Largest single-arm share of the population (1.0 = everyone on one arm).
    pub max_share: f64,
    /// Herfindahl concentration `sum rho_a^2` (1/|A| = perfectly spread, 1 = one arm).
    pub hhi: f64,
    /// p95 of realized latency = base * (1 + c(rho_assigned)) over prompts.
    pub p95_latency_ms: f64,
    pub mean_latency_ms: f64,
    pub picks: BTreeMap<String, usize>,
}

/// Population shares `rho_a = n_a / N` of an assignment, as a load field.
fn shares_of(assign: &[String], n: usize) -> LoadField {
    let n = n.max(1) as f64;
    let mut picks: BTreeMap<String, usize> = BTreeMap::new();
    for a in assign {
        *picks.entry(a.clone()).or_insert(0) += 1;
    }
    picks.into_iter().map(|(a, c)| (a, c as f64 / n)).collect()
}

/// Score a fixed assignment, charging latency congestion — and reporting concentration —
/// at the load field `load`. For an equilibrium `load` is the converged flow `rho*`
/// (which is the physically meaningful population split; at a *mixed* equilibrium the
/// pure last-iterate assignment would over-count one arm, so concentration is read from
/// `rho*`, consistent with the congestion charge). For a one-shot policy `load` is the
/// assignment's own shares (see [`measure`]), so the two coincide. Realized quality is
/// read per prompt from its assigned arm.
fn measure_at(sims: &[SimPrompt], assign: &[String], load: &LoadField) -> Outcome {
    let n = sims.len();
    let (mut q, mut lat_sum) = (0.0, 0.0);
    let mut lats = Vec::with_capacity(n);
    let mut picks: BTreeMap<String, usize> = BTreeMap::new();
    for (s, a) in sims.iter().zip(assign) {
        *picks.entry(a.clone()).or_insert(0) += 1;
        let c = &s.cells[a];
        q += c.quality;
        let eff = c.latency_ms * (1.0 + congestion(load.get(a).copied().unwrap_or(0.0)));
        lat_sum += eff;
        lats.push(eff);
    }
    lats.sort_by(|a, b| a.total_cmp(b));
    let p95 = if lats.is_empty() {
        0.0
    } else {
        lats[((lats.len() as f64 * 0.95).ceil() as usize).min(lats.len()) - 1]
    };
    let denom = n.max(1) as f64;
    // Concentration = the equilibrium load distribution (sums to ~1).
    let max_share = load.values().copied().fold(0.0, f64::max);
    let hhi: f64 = load.values().map(|r| r * r).sum();
    Outcome {
        quality: q / denom,
        max_share,
        hhi,
        p95_latency_ms: p95,
        mean_latency_ms: lat_sum / denom,
        picks,
    }
}

/// A one-shot policy's outcome: latency congestion charged at the assignment's own
/// induced load (each request pays the load its own policy creates).
fn measure(sims: &[SimPrompt], assign: &[String]) -> Outcome {
    measure_at(sims, assign, &shares_of(assign, sims.len()))
}

/// Best response: each prompt's load-aware argmax at the current load field.
fn best_response(heads: &Heads, sims: &[SimPrompt], slo: &Slo, rho: &LoadField, gamma: f64) -> Vec<String> {
    sims.iter()
        .map(|s| {
            heads
                .best_field(&s.x, slo, rho, gamma)
                .map(|(m, _)| m)
                .unwrap_or_default()
        })
        .collect()
}

/// Damping factor for the load update (fictitious play): the new load field is a convex
/// blend of the old and the best-response target. Sub-1 damping is what makes the
/// iteration converge instead of oscillating between corners in a congestion game.
const DAMP: f64 = 0.3;
const RHO_TOL: f64 = 1e-4;
const MAX_ITERS: usize = 500;

/// Solve the Wardrop fixed point for the load-aware head at field strength `gamma` by
/// damped fictitious play: best-respond to the current load, blend the induced load in,
/// repeat until the *load field* converges, then measure the equilibrium assignment.
/// Converging on `rho` (not the raw assignment) is what tames the corner-oscillation a
/// pure best response suffers. `gamma = 0` is the static head — its pick is
/// load-independent, so the first best response is already the fixed point.
pub fn equilibrium(heads: &Heads, sims: &[SimPrompt], slo: &Slo, gamma: f64) -> Outcome {
    let arms: Vec<String> = {
        let mut a: Vec<String> = sims.iter().flat_map(|s| s.cells.keys().cloned()).collect();
        a.sort();
        a.dedup();
        a
    };
    let uniform = 1.0 / arms.len().max(1) as f64;
    let mut rho: LoadField = arms.iter().map(|a| (a.clone(), uniform)).collect();
    for _ in 0..MAX_ITERS {
        let target = shares_of(&best_response(heads, sims, slo, &rho, gamma), sims.len());
        let mut delta = 0.0f64;
        for a in &arms {
            let cur = rho.entry(a.clone()).or_insert(0.0);
            let next = (1.0 - DAMP) * *cur + DAMP * target.get(a).copied().unwrap_or(0.0);
            delta = delta.max((next - *cur).abs());
            *cur = next;
        }
        if delta < RHO_TOL {
            break; // load field converged
        }
    }
    // Equilibrium assignment at the converged load. Latency congestion is charged at the
    // equilibrium load `rho` (the smoothed load the population actually experiences), so
    // the tail is a stable fixed-point quantity, not a knife-edge last-iterate artifact.
    let assign = best_response(heads, sims, slo, &rho, gamma);
    measure_at(sims, &assign, &rho)
}

/// The oracle: assign each prompt its own highest-quality arm (the realized-quality
/// ceiling). Its latency is measured under the load *it* induces — so a high-quality
/// oracle that happens to concentrate still pays a tail.
pub fn oracle(sims: &[SimPrompt]) -> Outcome {
    let assign: Vec<String> = sims
        .iter()
        .map(|s| {
            s.cells
                .iter()
                .max_by(|a, b| a.1.quality.total_cmp(&b.1.quality))
                .map(|(m, _)| m.clone())
                .unwrap_or_default()
        })
        .collect();
    measure(sims, &assign)
}

/// The best-single reference: everyone to `arm` (max concentration). `arm` is the
/// train-best arm — the pathology a static quality-only router degenerates toward.
pub fn best_single(sims: &[SimPrompt], arm: &str) -> Outcome {
    let assign = vec![arm.to_string(); sims.len()];
    measure(sims, &assign)
}

/// Mean and sample-std of a column across seeds.
pub fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = if xs.len() > 1 {
        xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    (m, v.sqrt())
}

/// Train-best arm by mean quality over `train` prompts.
pub fn train_best_arm(prompts: &[Prompt], train: &[usize]) -> String {
    let mut sum: BTreeMap<String, (f64, u32)> = BTreeMap::new();
    for &i in train {
        for (a, &q) in &prompts[i].arm_quality {
            let e = sum.entry(a.clone()).or_insert((0.0, 0));
            e.0 += q;
            e.1 += 1;
        }
    }
    sum.into_iter()
        .max_by(|x, y| (x.1 .0 / x.1 .1 as f64).total_cmp(&(y.1 .0 / y.1 .1 as f64)))
        .map(|(a, _)| a)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slo() -> Slo {
        Slo {
            lambda_cost: 0.0,
            mu_latency: 0.0,
            ..Slo::default()
        }
    }

    // Two arms: "fast" (lower quality, low latency) and "slow" (higher quality, high
    // latency) — the real corpus's shape (gpt-5.5 = best AND slowest). Prompts are
    // HETEROGENEOUS: each has a different quality margin (via x[General]), so there is a
    // smooth interior Wardrop equilibrium (identical prompts would give a degenerate
    // all-or-nothing corner). A quality-only static head still sends every prompt to
    // "slow" (it always has the higher quality); a field must spread the low-margin
    // prompts off the congested "slow" arm.
    fn two_arm_sims(n: usize) -> (Heads, Vec<SimPrompt>) {
        use hanzo_router::featurize::{FEAT_DIM, NUM_TASKS};
        use hanzo_router::registry::Task;
        let g = Task::General.index();
        let k = NUM_TASKS + 4;
        let mut w = vec![0.0; FEAT_DIM * k];
        w[g * k + g] = 1.0; // utility = x[General] * p[General]
        let mkarm = |m: &str, q: f64| {
            let mut f = vec![0.0; k];
            f[g] = q;
            Arm { model: m.into(), feat: f }
        };
        let heads = Heads::new(w, vec![mkarm("fast", 0.6), mkarm("slow", 0.9)]);
        let sims: Vec<SimPrompt> = (0..n)
            .map(|i| {
                let mut x = vec![0.0; FEAT_DIM];
                x[g] = 0.2 + 0.8 * (i as f64 / n as f64); // varied margin per prompt
                SimPrompt {
                    x,
                    cells: [
                        ("fast".into(), Cell { quality: 0.6, latency_ms: 1000.0, cost: 1.0 }),
                        ("slow".into(), Cell { quality: 0.9, latency_ms: 8000.0, cost: 1.0 }),
                    ]
                    .into(),
                }
            })
            .collect();
        (heads, sims)
    }

    #[test]
    fn static_concentrates_and_field_spreads_cutting_the_tail() {
        let (heads, sims) = two_arm_sims(120);
        let stat = equilibrium(&heads, &sims, &slo(), 0.0);
        assert_eq!(stat.max_share, 1.0, "static quality-only piles onto the best arm");
        // A field spreads the low-margin load off the congested best arm.
        let field = equilibrium(&heads, &sims, &slo(), 0.3);
        assert!(
            field.max_share < stat.max_share,
            "load-aware routing must reduce concentration ({} !< {})",
            field.max_share,
            stat.max_share
        );
        assert!(
            field.p95_latency_ms < stat.p95_latency_ms,
            "spreading load must cut the congested tail ({} !< {})",
            field.p95_latency_ms,
            stat.p95_latency_ms
        );
        // ...at a bounded quality cost (the low-margin prompts were nearly ties).
        assert!(
            field.quality >= stat.quality - 0.15,
            "quality must not collapse ({} vs {})",
            field.quality,
            stat.quality
        );
    }

    #[test]
    fn field_zero_equals_static_pick() {
        let (heads, sims) = two_arm_sims(20);
        let a = equilibrium(&heads, &sims, &slo(), 0.0);
        // gamma=0 is load-independent: every prompt still picks the higher-quality arm.
        assert_eq!(a.picks.get("slow").copied().unwrap_or(0), 20);
    }
}
