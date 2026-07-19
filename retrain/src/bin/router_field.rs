//! `router-field` --- the mean-field (load-aware) routing proof as a runnable report.
//!
//! Reads the augmented per-arm counterfactual JSONL (the same corpus `router-decide`
//! uses), fits the learned head on train prompts, then simulates three policies over
//! held-out prompts under a congestion model and reports, across the field strength
//! `gamma`: realized quality, hot-arm concentration (max share + Herfindahl), and a p95
//! tail-latency proxy --- mean +- std over prompt-level split seeds. Nothing is fit to
//! the holdout. The claim (hold quality, cut concentration + tail) stands or falls on
//! these numbers; a load-aware policy that costs quality with no concentration win is a
//! clean refutation and prints as one.

use anyhow::{Context, Result};
use clap::Parser;

use router_retrain::decision::{parse_labeled, pure_slo, split_by_prompt, to_prompts};
use router_retrain::field::{
    best_single, build, equilibrium, fit_head, mean_std, oracle, train_best_arm, Outcome,
};

#[derive(Parser, Debug)]
#[command(name = "router-field", about = "Mean-field (load-aware) routing proof.")]
struct Args {
    /// Augmented EvalSample JSONL (benchmark + id + EvalSample fields per line).
    #[arg(long)]
    events: std::path::PathBuf,
    /// Holdout fraction per benchmark (the simulated population).
    #[arg(long, default_value_t = 0.25)]
    holdout: f64,
    /// Comma-separated mean-field strengths gamma to sweep (0 = static head).
    #[arg(long, default_value = "0,0.02,0.05,0.1,0.2,0.5,1.0")]
    gammas: String,
    /// Ridge strength of the head fit (the shipped bundle uses 0.1).
    #[arg(long, default_value_t = 0.1)]
    ridge: f64,
    /// Comma-separated seeds to average the split over.
    #[arg(long, default_value = "7,8,9,10,11,12,13,14")]
    seeds: String,
}

fn parse_f(s: &str) -> Vec<f64> {
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}
fn parse_u(s: &str) -> Vec<u64> {
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}

/// Mean +- std of one metric extracted from a per-seed outcome vector.
fn col(all: &[Outcome], f: fn(&Outcome) -> f64) -> (f64, f64) {
    mean_std(&all.iter().map(f).collect::<Vec<_>>())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let text = std::fs::read_to_string(&args.events)
        .with_context(|| format!("read {}", args.events.display()))?;
    let prompts = to_prompts(&parse_labeled(&text)?);
    let gammas = parse_f(&args.gammas);
    let seeds = parse_u(&args.seeds);
    let slo = pure_slo(); // pure-quality SLO: the mean-field term is the ONLY latency-awareness.

    let n_bench = {
        let mut b: Vec<&str> = prompts.iter().map(|p| p.benchmark.as_str()).collect();
        b.sort();
        b.dedup();
        b.iter()
            .map(|name| {
                format!(
                    "{name}={}",
                    prompts.iter().filter(|p| &p.benchmark == name).count()
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!(
        "events={} prompts={} [{}] holdout={} seeds={} ridge={}",
        parse_labeled(&text)?.len(),
        prompts.len(),
        n_bench,
        args.holdout,
        seeds.len(),
        args.ridge
    );
    println!("model: rho_a = n_a/N ; congestion c(rho)=rho/(1-rho) ; eff_latency = base*(1+c) ; pure-quality SLO\n");

    // Per-seed: fit head on train, simulate the held-out population.
    let mut per_gamma: Vec<(f64, Vec<Outcome>)> = gammas.iter().map(|&g| (g, Vec::new())).collect();
    let mut oracles = Vec::new();
    let mut bests = Vec::new();
    let mut best_arm = String::new();
    for &seed in &seeds {
        let (tr, te) = split_by_prompt(&prompts, args.holdout, seed);
        let head = fit_head(&prompts, &tr, args.ridge);
        let (sims, _arms) = build(&prompts, &te);
        for (g, out) in per_gamma.iter_mut() {
            out.push(equilibrium(&head, &sims, &slo, *g));
        }
        oracles.push(oracle(&sims));
        best_arm = train_best_arm(&prompts, &tr);
        bests.push(best_single(&sims, &best_arm));
    }

    // Absolute table: quality, concentration, tail, per gamma.
    println!("--- policies (mean +- std over seeds) ---");
    println!(
        "policy            quality          max_share        HHI              p95_latency_ms      mean_latency_ms"
    );
    let row = |name: &str, all: &[Outcome]| {
        let (q, qs) = col(all, |o| o.quality);
        let (ms, mss) = col(all, |o| o.max_share);
        let (h, hs) = col(all, |o| o.hhi);
        let (p, ps) = col(all, |o| o.p95_latency_ms);
        let (m, msd) = col(all, |o| o.mean_latency_ms);
        println!(
            "{name:<16}  {q:.4}+-{qs:.4}  {ms:.4}+-{mss:.4}  {h:.4}+-{hs:.4}  {p:8.0}+-{ps:<7.0}  {m:8.0}+-{msd:.0}"
        );
    };
    for (g, all) in &per_gamma {
        let name = if *g == 0.0 {
            "static (g=0)".to_string()
        } else {
            format!("field g={g}")
        };
        row(&name, all);
    }
    row(&format!("best-single[{best_arm}]"), &bests);
    row("oracle", &oracles);

    // The tradeoff vs the static head, paired on identical holdouts each seed.
    let static_out = &per_gamma[0].1;
    println!(
        "\n--- tradeoff vs static head (paired per seed; d<0 = drop, d>0 = gain) ---"
    );
    println!("gamma   d_quality           d_max_share         d_p95_latency_ms     quality_retained");
    for (g, all) in &per_gamma {
        if *g == 0.0 {
            continue;
        }
        let dq: Vec<f64> = all
            .iter()
            .zip(static_out)
            .map(|(a, s)| a.quality - s.quality)
            .collect();
        let dc: Vec<f64> = all
            .iter()
            .zip(static_out)
            .map(|(a, s)| a.max_share - s.max_share)
            .collect();
        let dp: Vec<f64> = all
            .iter()
            .zip(static_out)
            .map(|(a, s)| a.p95_latency_ms - s.p95_latency_ms)
            .collect();
        let (mq, sq) = mean_std(&dq);
        let (mc, sc) = mean_std(&dc);
        let (mp, sp) = mean_std(&dp);
        let (sq_mean, _) = col(static_out, |o| o.quality);
        let (fq_mean, _) = col(all, |o| o.quality);
        println!(
            "{g:<7} {mq:+.4}+-{sq:.4}   {mc:+.4}+-{sc:.4}   {mp:+8.0}+-{sp:<7.0}  {:.1}%",
            100.0 * fq_mean / sq_mean.max(1e-9)
        );
    }
    println!(
        "\nn.b. quality_retained = field quality / static quality. A row with d_max_share<0 and \
         d_p95<0 at d_quality~=0 confirms the mean-field claim; d_quality<0 with d_max_share~=0 refutes it."
    );
    Ok(())
}
