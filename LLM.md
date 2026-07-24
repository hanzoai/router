# LLM.md — hanzoai/router

Guidance for AI agents working in this repo.

## What this is
`hanzo-router` — the **pure-Rust model-routing core**. Value-in / decision-out:
classify a request, then pick from a pool of local + cloud models. No hardware
access of its own → deterministic, unit-testable. Optional `proxy` feature adds an
axum replica load-balancing front (prefix-affinity + least-loaded + health probe).

## Canonical role
Real code lives here (canonical impl). It is the routing brain that `hanzoai/node`
uses with `hanzoai/ml` + `hanzoai/engine`. This is Rust ecosystem infra, not an SDK
wrapper — completeness order across langs is Python → Rust → C++ → Go. Discovery/
wrapper repos link OUT to this; never duplicate the impl. One impl, one place (DRY).

## Build / run
- `cargo test` — pure core. `cargo test --features proxy` — + the proxy front.
- Proxy binary: `cargo run --features proxy --bin hanzo-router -- --model M --replica URL ...`
- Learned heads: `cargo run -p router-retrain --bin router-fit -- --events E.jsonl --out H.safetensors`

## Key entry points
- `src/lib.rs` — public API (`route`, `load_policy`), module map.
- `src/policy.rs` — `Policy::select` (Reuse → LoadLocal → Cloud → NoFit).
- `src/route.rs` — `RoutePolicy` seam (mechanism vs. learned brain).
- `src/proxy.rs` + `src/bin/proxy.rs` — the replica proxy front.
- `learner/` (enso brain) · `retrain/` (router-fit + eval) · `heads/` (serve bundle).

## Brand rules (hard — enforce in all docs)
- **Never** call Hanzo an "LLM gateway" or position vs LiteLLM. It is a routing core
  within **Hanzo — the Open AI Cloud**, not a proxy product.
- Paths are `/v1/...` only — never `/api/`.
- Zen models are our own family; never name upstream models in public copy.

Full model: `~/work/hanzo/SDK-ARCHITECTURE.md`.
