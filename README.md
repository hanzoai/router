# Hanzo Router

**Pure-Rust model-routing core** — memory- and engine-aware, pure logic over value inputs.

`hanzo-router` is the native routing brain that [`hanzo-node`](https://github.com/hanzoai/node)
uses (alongside [`hanzo-ml`](https://github.com/hanzoai/ml) and
[`hanzo-engine`](https://github.com/hanzoai/engine)) to route each request across a
swappable pool of **local + cloud** models. It is **pure logic over value inputs**
(Rich Hickey style): the caller hands it a memory snapshot, a registry of available
models, and the set currently loaded — the router decides, with **no hardware access
of its own**, so every decision is deterministic and unit-testable.

Part of **Hanzo — the Open AI Cloud**.

## What it decides

Given a classified request and the runtime, `Policy::select` prefers, in order:

1. **Reuse** — a model already loaded in a connectable engine (zero load cost).
2. **LoadLocal** — a local model whose resident footprint *fits* available memory
   (the "run it locally if the RAM is there" path).
3. **Cloud** — fall through to the cheapest usable provider.
4. **NoFit** — nothing usable; the caller decides how to degrade.

Vision, context length, and per-request SLO (cost / latency / quality ceilings) gate
the choice before it is made.

## Install

```toml
[dependencies]
hanzo-router = { git = "https://github.com/hanzoai/router" }

# with the replica load-balancing proxy front:
hanzo-router = { git = "https://github.com/hanzoai/router", features = ["proxy"] }
```

The pure router core has a tiny dependency set (`serde` only). The async/HTTP stack
is pulled in **only** under the `proxy` feature, so the decision core stays light.

## Quickstart

```rust
use hanzo_router::{
    classify::{Heuristic, Request, Classifier},
    memory::MemSnapshot,
    policy::{Policy, Context, Decision},
    registry::{Registry, ModelCard, Backend, Task},
};
use std::collections::BTreeSet;

let registry = Registry::new(vec![
    ModelCard {
        id: "deepseek-v4-flash".into(),
        backend: Backend::Local { est_bytes: 93 << 30 },
        tasks: vec![Task::Code, Task::Reasoning, Task::General],
        max_context: 1_048_576, vision: false, cost_per_1k: 0.0,
    },
    ModelCard {
        id: "claude-sonnet-4-5".into(),
        backend: Backend::Cloud { provider: "anthropic".into() },
        tasks: vec![Task::Code, Task::General],
        max_context: 200_000, vision: true, cost_per_1k: 3.0,
    },
]);

let policy = Policy::default();
let task = Heuristic.classify(&Request { text: "fix this ```rust``` bug".into(), ..Default::default() });

// 121GB unified box, ~115GB free -> V4 (93GB) fits and is preferred for code.
let mem = MemSnapshot { available_bytes: 115 << 30, total_bytes: 121 << 30, unified: true };
let ctx = Context {
    task, registry: &registry, mem, running: &BTreeSet::new(),
    vision_required: false, min_context: 0,
};
assert!(matches!(policy.select(&ctx), Decision::LoadLocal { .. }));
```

Or route in one shot with the top-level `hanzo_router::route(...)` helper: classify,
then select.

## Mechanism vs. brain

The routing decision is a seam: one trait, `RoutePolicy`, turns a request into a
`Route` (which model, at what level, in what modality, with what confidence). This
decomplects mechanism from brain:

- **Mechanism** — `hanzo-router` owns the registry, the SLO gate, placement, dispatch,
  and the safety-refusal sentinel.
- **Brain** — the policy that picks. Two implementations, one `Policy` type:
  - **Rule-based** `Policy` (declarative `prefer:` YAML) — the cold-start fallback, so a
    deployment routes sensibly *before* any eval data exists.
  - **Learned heads** — a ridge-fit policy loaded from a serve bundle. The engine mounts
    it via `ROUTER_HEADS=<path>` (`Policy::load_heads` at startup); unset falls back to
    the rule-based policy.

Load a policy from YAML:

```rust
let policy = hanzo_router::load_policy("prefer:\n  cheap_chat: [gpt-4o-mini]\n")?;
```

## Learned routing heads

A learned policy is a serve bundle of ridge-fit heads (`src/heads.rs`: one matrix `W`
and the arm profiles). The engine mounts one with `ROUTER_HEADS=<path>`
(`Policy::load_heads`); unset, routing uses the rule-based policy.

## Load-aware routing (mean-field)

The optional load field (`ROUTER_FIELD`, **default OFF**, fully reversible) makes routing
congestion-aware. On the real counterfactual corpus (2,984 events / 373 prompts, 8 arms),
load-awareness cuts hot-arm concentration and tail latency at ~zero quality cost — the
Wardrop-equilibrium claim holds.
Recommended deployment `gamma` in `[0.05, 0.2]`.

## Replica proxy (`proxy` feature)

Under `--features proxy` the crate ships the `hanzo-router` binary: an axum reverse
proxy that fronts N engine replicas serving the same model. It picks a replica per
request — **prefix-affinity** (a conversation sticks to one replica) + **least-loaded**
spill — streams the response back byte-for-byte (SSE passes through unchanged), and a
background loop re-probes replica `/health` to evict and auto-restore.

```bash
hanzo-router --host 0.0.0.0 --port 1234 \
  --model qwen3-30b-a3b --upstream-model default \
  --replica http://10.0.0.1:8080 \
  --replica http://10.0.0.2:8080
```

Admin surface (all `/v1`): `GET /v1/replicas` (health + in-flight), `POST /v1/replicas`
(register a replica), `GET /health`. A YAML pool file works too (`--config pool.yaml`).

**Why `--upstream-model`**: a single-model engine strict-matches its served id (or
`default`) and returns HTTP 500 on anything else. Clients that send `claude-*`-style
ids are pinned to the served id on the forwarded request, while routing/affinity still
key on the client's *original* model.

## Replica fabric — deployment runbook

A worked deployment: 3 independent full replicas of one MoE model, one per box,
load-balanced behind `hanzo-router` so N parallel agents each run at full GPU speed.

**Front endpoint** (what a client points at):

```bash
export ANTHROPIC_BASE_URL=http://10.0.0.144:1234   # the router
export ANTHROPIC_API_KEY=local                      # engine ignores auth (dev)
hanzo code
```

OpenAI clients use `http://10.0.0.144:1234/v1`. Both `/v1/messages` (Anthropic) and
`/v1/chat/completions` (OpenAI) forward through, SSE streamed byte-for-byte.

**Per-box serve** (native `hanzo serve`, each binds `:8080`, served model id = `default`):
config + tokenizer live in a local `qwen3-30b-meta/` dir (config.json,
generation_config.json, tokenizer.json, tokenizer_config.json + a symlink to the gguf)
so `-m <dir>` loads locally with no HuggingFace fetch.

```bash
# CUDA box
hanzo serve -m Qwen/Qwen3-30B-A3B-Instruct-2507 --format gguf \
  -f ~/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf -p 8080 --host 0.0.0.0 -n 0:48

# ROCm APU (gfx1151) — ONE ROCm process only (concurrent kfd init wedges):
LD_LIBRARY_PATH=/opt/rocm/lib hanzo serve \
  -m ~/models/qwen3-30b-meta --format gguf -f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  -p 8080 --host 127.0.0.1 -n 0:48 --pa-context-len 131072
# -n 0:48 forces all 48 layers onto the GPU (the auto-mapper misreads unified APU
#   memory and offloads to CPU -> a 4 tok/s trap).
# --pa-context-len 131072 caps KV to ~24GB (the default grabs ~81GB).

# Metal (M-series)
hanzo serve -m ~/models/qwen3-30b-meta --format gguf \
  -f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf -p 8080 --host 0.0.0.0 \
  -n 0:48 --paged-attn on --pa-context-len 131072
# paged-attn is auto-OFF on Metal; force `on` so KV is bounded + block-shared.
```

**Router** (on any box that can reach the replicas):

```bash
hanzo-router --host 0.0.0.0 --port 1234 \
  --model qwen3-30b-a3b --upstream-model default \
  --replica http://192.168.77.2:8080 \
  --replica http://127.0.0.1:8080 \
  --replica http://10.0.0.132:8080
```

**Measured** (Qwen3-30B-A3B Q4_K_M, 256-tok decode): single box ~48–72 tok/s depending on
backend; through the router single-stream 58–63 tok/s; **3 concurrent agents (1 per box)
~186 tok/s aggregate, each 62–68 tok/s** — the 3× parallel win. Beyond 3, agents double up
per box and per-agent decode drops (MoE reads distinct experts per sequence). Sweet spot:
≤3 concurrent, one full-speed agent per GPU. Spreading across boxes is the design; the
router does exactly this.

> ROCm/gfx1151 caveat: on borderline prompts the first generated token's argmax can flip
> (a HIP MoE-topk "last-row collapse" — the same class fixed on the CUDA path). Realistic
> agent/code prompts are clean; to exclude an affected box, drop its `--replica`.

## Workspace

| Crate | Role |
|-------|------|
| `hanzo-router` (root) | the mechanism — registry, SLO gate, policy seam, heads loader, proxy front |

## Tests

```bash
cargo test                    # pure router core
cargo test --features proxy   # + the reverse-proxy front (mock-engine e2e)
```

The proxy suite drives two mock engines over real HTTP and proves prefix-affinity
stickiness, streaming passthrough with in-flight lease release, and probe-driven health
eviction + auto-restore.

## Hanzo — the Open AI Cloud

Open source · every language · on-chain settlement. [hanzo.ai](https://hanzo.ai) · [docs.hanzo.ai](https://docs.hanzo.ai)

**SDKs in every language** — [Python](https://github.com/hanzoai/python-sdk) (flagship) · [TypeScript](https://github.com/hanzo-js/sdk) · [Go](https://github.com/hanzo-go/sdk) · [Rust](https://github.com/hanzo-rs/sdk) · [C++](https://github.com/hanzo-cpp/sdk) · [Swift](https://github.com/hanzo-swift/sdk) · [Kotlin](https://github.com/hanzo-kt/sdk) · [umbrella](https://github.com/hanzoai/sdk)
