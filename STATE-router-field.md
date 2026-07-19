# Router mean-field — resume state (agent hit session cap 9:20pm; CTO checkpointed)

## DONE + verified (branch wip/router-mean-field, builds clean, 49/50 tests green)
- Load-aware scoring wired: `Policy.field_gamma: Option<f64>` (None == static head, reversible);
  `field_gamma_from_env()` (ROUTER_FIELD); congestion term in heads.rs scoring; replica.rs load feed.
- field.rs: `equilibrium()` simulator (per-arm counterfactual → realized quality + concentration +
  p95-latency-proxy under a congestion load model); `router_field` bin (--events, gamma sweep).
- CTO fixes on takeover: bench_overhead.rs Policy init (+field_gamma: None); the synthetic 2-arm
  unit test #[ignore]'d with reason (authoring pass flagged its arm-gap as unrealistic — do NOT
  tune-to-green; tighten the synthetic arms to realistic closeness instead).

## REMAINING (the science — authoring pass reported it STRONG but the CTO could NOT reproduce: the
## regenerated events corpus was cleaned from scratchpad)
1. Regenerate the augmented counterfactual events JSONL from ~/work/hanzo/enso-bench-data/results
   (the id→prompt-text join + per-arm augmentation the router-fit/router-decide path builds; the
   earlier 2,981-event samples_pooled.jsonl is gone — rebuild it).
2. Run `cargo run -p router-retrain --bin router_field -- --events <that.jsonl>` → the 3-policy
   proof (static vs field vs oracle): realized quality (must hold ~0.9215), max-arm concentration
   (must drop), p95 latency proxy vs gamma. Commit the writeup.
3. Fix the synthetic fixture (realistic arm closeness) and un-#[ignore] the unit test.
4. Ship to hanzoai/router main (ROUTER_FIELD default OFF = safe). enso-bench roofline continual
   registration (additive) still pending.

## The claim under test: load-aware routing holds quality while cutting hot-arm concentration + tail
## (Wardrop equilibrium). NOT yet CTO-reproduced end-to-end — do not claim proven until the corpus run.
