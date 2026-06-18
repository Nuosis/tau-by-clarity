# Headroom Probe Findings

Status: behavior probe only. Tau is not wired to Headroom here.

## Current Tau Boundary

`pi_coding_agent.active_compression` is Tau's legacy local CCR/compression
implementation. It is not the Headroom SDK path.

Do not tune that package as if it were Headroom. Keep it isolated until the real
Headroom integration boundary is chosen.

## Probe Command

```bash
/Users/marcusswift/cli/tau-by-clarity/.venv/bin/python \
  packages/coding-agent/evals/ccr_query_guidance/headroom_probe.py
```

The probe bypasses Tau's local compressor and checks real Headroom behavior:

1. `headroom.compress()` against hard CCR scenario payloads.
2. `headroom.cache.compression_store.CompressionStore.search()` against the same
   originals.

## Observed Behavior

Public `headroom.compress()` preserved all expected evidence terms, but did so
because it saved `0` tokens on every hard scenario. In the realistic
tool-call/tool-result/follow-up message shape:

- `read` tool results were excluded/protected.
- `bash` tool results were also not compressed in these scenario shapes.
- All public-compress rows reported `saved=0`.

This means the public SDK path did not reproduce Tau's local CCR compression
behavior. It avoided data loss by not compressing these payloads.

Headroom CCR search did not pass all hard scenarios at the default
`score_threshold=0.3`. Missing expected terms:

- `json_order_lookup`
- `entity_record_lookup`
- `trace_span_lookup`
- `config_key_lookup`
- `api_error_code_lookup`
- `mixed_noise_lookup`
- `incident_trace_chain`
- `claim_denial_chain`

Lowering the threshold helps some cases but does not fully solve underscore-heavy
or multi-hop hard cases:

- `0.05` still missed `config_key_lookup`, `api_error_code_lookup`, and part of
  `incident_trace_chain`.
- `0.3` misses many exact-ID/hyphen/underscore cases.

## Implication

Real Headroom is not a drop-in proof that hard `ccr_retrieve` query workflows are
solved for Tau.

Before integration, we need a concrete Headroom boundary decision:

1. Use Headroom public compression only, accepting that many fresh tool outputs
   may remain uncompressed.
2. Use Headroom lower-level `ContentRouter`/`CompressionStore` APIs and add
   Tau-specific retrieval/search support where Headroom search misses hard cases.
3. Keep Tau legacy CCR temporarily behind a clearly named legacy backend while
   Headroom integration is built and compared.

No publish decision should be based on this probe alone.
