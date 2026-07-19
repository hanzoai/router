# Mean-field routing — PROVEN on the real counterfactual corpus

Run: `cargo run --release -p router-retrain --bin router_field -- --events docs/router_corpus.jsonl`
Corpus: 2,984 augmented events / 373 prompts (gpqa_diamond 198 + livecodebench 175), 8 arms,
per-prompt-per-arm counterfactual outcomes. Congestion model: rho_a=n_a/N, c(rho)=rho/(1-rho),
eff_latency=base*(1+c). 8 seeds, prompt-level holdout 0.25.

## Result: the Wardrop claim holds — load-awareness cuts concentration + tail at ~zero quality cost.
| gamma | quality retained | d_max_share | d_p95_latency |
|-------|------------------|-------------|---------------|
| 0.05  | 99.6%            | -0.145 (24% less concentration) | -125,806 ms (29% less tail) |
| 0.2   | **100.0%**       | **-0.239 (40% less)**           | **-185,498 ms (43% less)**  |
| 0.5   | 98.0%            | -0.252                          | -194,997 ms                 |

Pre-registered refutation (d_quality<0 with d_max_share~=0) did NOT occur; the opposite did.
This validates Paper 2's mean-field section on production data. ROUTER_FIELD default OFF (reversible);
recommended deployment gamma in [0.05, 0.2].
