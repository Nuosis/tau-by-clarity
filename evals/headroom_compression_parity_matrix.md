# Headroom Compression Parity Matrix

Scope: compression platform behavior only. This excludes Headroom SDKs,
MCP servers, provider proxy routing, dashboard SQL, billing enforcement, and
output-shaper A/B accounting unless the behavior directly protects or reports
input compression.

Last verified in this worktree:

```bash
PYTHONPATH=packages/ai/src:packages/coding-agent/src:packages/agent/src:packages/tui/src \
uv run python evals/headroom_compression_parity.py /tmp/headroom-src
# all 208 fixtures passed

# 2026-06-19 focused re-check:
# all 208 fixtures passed; aggregate char savings 62.9%

PYTHONPATH=packages/ai/src:packages/coding-agent/src:packages/agent/src:packages/tui/src \
uv run pytest packages/ai/tests/test_compression_policy.py \
  packages/coding-agent/tests/test_active_compression_routes.py \
  packages/coding-agent/tests/test_active_compression_extension.py \
  packages/coding-agent/tests/test_bash_active_compression.py \
  packages/coding-agent/tests/test_core_utils.py::TestCliDebugLog \
  packages/ai/tests/test_pii_hook.py packages/ai/tests/test_stream.py \
  packages/ai/tests/test_tokenization.py -q
# 171 passed

# 2026-06-19 focused active-compression re-check:
# packages/ai/tests/test_compression_policy.py
# packages/coding-agent/tests/test_active_compression_routes.py
# packages/coding-agent/tests/test_active_compression_extension.py
# packages/coding-agent/tests/test_bash_active_compression.py
# packages/ai/tests/test_tokenization.py
# 84 passed
```

## Requirement Matrix

| Headroom compression area | Source of requirement | Tau evidence | Status |
| --- | --- | --- | --- |
| SmartCrusher-style structured JSON compression | `/tmp/headroom-src/tests/parity/fixtures/smart_crusher`; `/tmp/headroom-src/headroom/transforms/kompress_compressor.py` | `smart_crusher 17/17` in `evals/headroom_compression_parity.py`; route tests in `packages/coding-agent/tests/test_active_compression_routes.py` | Covered by fixtures and route tests |
| Log compression, warning/error preservation, template compaction | `/tmp/headroom-src/tests/parity/fixtures/log_compressor`; `/tmp/headroom-src/tests/test_log_compressor.py`; `/tmp/headroom-src/headroom/transforms/log_compressor.py` | `log_compressor 20/20`; route tests for warnings, errors, stack traces, summaries | Covered |
| Diff compression and unidiff/noise handling | `/tmp/headroom-src/tests/parity/fixtures/diff_compressor`; `/tmp/headroom-src/tests/test_transforms/test_diff_compressor.py` | `diff_compressor 27/27`; route tests for thresholds, binary diff, rename/copy metadata, hunk preservation | Covered |
| Search/code/HTML compression routes | `/tmp/headroom-src/tests/test_search_compressor.py`; `/tmp/headroom-src/tests/test_transforms_search_compressor.py`; `/tmp/headroom-src/tests/test_compression/test_code_handler.py` | `search_code_html 10/10`; content detector `21/21`; route tests for search caps, code signatures, HTML extraction | Covered |
| Universal compressor structure-preservation invariants | `/tmp/headroom-src/tests/test_compression/test_universal.py`; `/tmp/headroom-src/tests/test_compression/test_masks.py`; `/tmp/headroom-src/tests/test_compression/test_json_handler.py`; `/tmp/headroom-src/tests/test_compression/test_code_handler.py`; `/tmp/headroom-src/tests/test_compression/test_evals.py`; `/tmp/headroom-src/tests/test_compression/test_llm_eval.py` | `smart_crusher 17/17`; `search_code_html 10/10`; route tests for JSON object/tabular arrays/opaque blobs, code signatures/docstrings, and mixed content | Covered behaviorally; Tau does not expose Headroom's handler/mask classes as public API |
| Content detection | `/tmp/headroom-src/tests/parity/fixtures/content_detector`; `/tmp/headroom-src/headroom/transforms/content_detector.py` | `content_detector 21/21`; `test_content_detector_matches_headroom_route_fixtures` | Covered |
| Tokenizer behavior and token floor | `/tmp/headroom-src/tests/parity/fixtures/tokenizer`; `/tmp/headroom-src/headroom/tokenizer.py`; `/tmp/headroom-src/headroom/tokenizers/` | `tokenizer 40/40`; `token_floor 1/1`; `packages/ai/src/pi_ai/tokenization.py` | Covered |
| Compression summaries for row offload | `/tmp/headroom-src/tests/test_compression_summary.py`; `/tmp/headroom-src/headroom/transforms/compression_summary.py` | `compression_summary 1/1`; row offload summary/reversibility checks | Covered |
| CCR storage, retrieval, TTL, eviction, metadata, read lifecycle | `/tmp/headroom-src/tests/test_ccr*.py`; `/tmp/headroom-src/headroom/ccr/`; `/tmp/headroom-src/tests/test_transforms/test_read_lifecycle.py` | `ccr_recovery 1/1`, `ccr_store 3/3`, `read_lifecycle 6/6`; `packages/coding-agent/src/pi_coding_agent/active_compression/ccr.py` | Covered for Tau's query-scoped retrieval model |
| Compressed-output cache reuse, LRU eviction, and frozen-prefix stability | `/tmp/headroom-src/tests/test_compression_cache.py`; `/tmp/headroom-src/tests/test_token_headroom_mode.py`; `/tmp/headroom-src/headroom/cache/compression_cache.py` | `test_active_compression_reuses_cached_tool_output_across_requests`; `test_active_compression_cache_evicts_least_recently_used_tool_output`; `test_active_compression_cache_does_not_rewrite_frozen_prefix`; `/compression stats` cache counters | Covered for Tau's active hook; cache replay is scoped to tool-result text units and live suffixes |
| Marker pinning/preservation and already-compressed guards | `/tmp/headroom-src/tests/test_compression_units.py`; `/tmp/headroom-src/headroom/transforms/compression_units.py` | `marker_pinning 2/2`, `marker_preserving 1/1`; route/unit-cache checks | Covered |
| Provider/cache prefix protection | `/tmp/headroom-src/tests/test_cache_aligner_detector_only.py`; `/tmp/headroom-src/tests/test_cache/test_prefix_tracker.py`; cache-control tests | `cache_control_protection 2/2`; `cache_aligner 1/1`; `packages/ai/src/pi_ai/stream.py` frozen-prefix annotation; `packages/ai/src/pi_ai/compression.py` cache-zone guards | Covered |
| Detector-only cache aligner volatility signals | `/tmp/headroom-src/tests/test_cache_aligner_detector_only.py`; `/tmp/headroom-src/headroom/transforms/cache_aligner.py` | `cache_aligner 1/1`; `packages/ai/tests/test_compression_policy.py` verifies UUID/ISO/JWT/hex labels, no prompt mutation, subscription skip | Covered |
| Compression policy by auth mode and net-cost formula | `/tmp/headroom-src/tests/test_compression_policy.py`; `/tmp/headroom-src/tests/test_compression_policy_toin_gate.py`; `/tmp/headroom-src/headroom/transforms/compression_policy.py` | `compression_policy 6/6`; `packages/ai/src/pi_ai/compression_policy.py`; policy tests | Covered |
| TOIN/learning write gate | `/tmp/headroom-src/tests/test_compression_policy_toin_gate.py`; `/tmp/headroom-src/RUST_DEV.md` TOIN notes | `compression_policy` and `compression_command` cases; `get_compression_learning_stats()` | Covered as counters/read-only gate, not Headroom's full TOIN model |
| Failure handling, non-shrinking rejection, circuit breaker, inflation guard | `/tmp/headroom-src/tests/test_compress_failure.py`; `/tmp/headroom-src/tests/test_compression_safety_rails.py` | `non_shrinking_rejection 1/1`, `compression_failure 4/4`, `compression_circuit_breaker 4/4`, `inflation_guard 1/1` | Covered |
| Image compression, OCR substitution, image optimize disable, and image-log redaction | `/tmp/headroom-src/tests/test_image_compression.py`; `/tmp/headroom-src/tests/test_image_compressor.py`; `/tmp/headroom-src/tests/test_image_compression_decision.py`; `/tmp/headroom-src/tests/test_image_log_redaction.py`; `/tmp/headroom-src/headroom/image/` | `image_compression 6/6`, `image_log_redaction 3/3`; `compression_image_optimize=False` preserves image bytes/OCR while text compression still runs; unit outcome reason `image_optimize_disabled` | Covered for compression behavior; Headroom proxy headers/tag stamping remain out of scope |
| Custom/workflow tag protection | `/tmp/headroom-src/tests/test_tag_protector_invariant.py`; `/tmp/headroom-src/tests/test_transforms/test_tag_protector.py`; `/tmp/headroom-src/headroom/transforms/tag_protector.py` | `custom_tag_protection 3/3`; route tests for duplicate, nested, self-closing, collision, lost-placeholder discard | Covered |
| Compression unit batching, slot preservation, protected prompt roles, unit outcome taxonomy | `/tmp/headroom-src/tests/test_compression_units.py`; `/tmp/headroom-src/headroom/transforms/compression_units.py` | `unit_cache 1/1`; provider/block traversal and cache-zone tests in `packages/ai/tests/test_pii_hook.py`; marker preserving cases; `test_active_compression_records_unit_outcomes_for_applied_and_protected_units`; `get_unit_outcome_stats()` | Covered behaviorally; Tau intentionally has no public `CompressionUnit` class |
| `/compression` command observability/reset | Headroom `/stats` and transform telemetry notes in `/tmp/headroom-src/RUST_DEV.md` | `compression_command 1/1`; `/compression stats` reports compression, learning, cache-alignment, and unit outcome counters | Covered for Tau CLI |
| Output savings A/B estimator | `/tmp/headroom-src/tests/test_output_savings.py`; `/tmp/headroom-src/headroom/proxy/output_savings.py` | Not implemented | Out of compression-input scope unless Marcus asks for output-shaper accounting |
| Proxy CostTracker/budget accounting | `/tmp/headroom-src/tests/test_cost_tracker_counterfactual.py`; Headroom proxy server | Tau reports token/byte savings but no proxy budget ledger | Out of scope except token/byte savings already covered |
| Compression and image compression proxy decision value objects | `/tmp/headroom-src/tests/test_compression_decision.py`; `/tmp/headroom-src/tests/test_image_compression_decision.py`; `/tmp/headroom-src/tests/test_handler_outcome_tag_invariant.py` | Tau uses direct context controls (`compression_image_optimize`, policy/auth mode, frozen-prefix/cache-zone guards) and unit outcomes instead of proxy headers/tags | Compression behavior covered; frozen dataclass/tag/dashboard contracts are proxy implementation details |
| Proxy headers, dashboards, SQL, provider server endpoints | `/tmp/headroom-src/tests/test_proxy_*`; `/tmp/headroom-src/sql/`; `/tmp/headroom-src/tests/test_proxy_compression_headers.py`; `/tmp/headroom-src/tests/test_proxy_compression_executor.py` | Not implemented | Out of scope per compression-only instruction |

## Scope Notes

1. Tau does not expose Headroom's `CompressionUnit` class as a public API.
   That is intentional under the compression-only scope; the behavior and
   reason/category observability are covered through the central hook,
   `get_unit_outcome_stats()`, `/compression stats`, and parity tests.
2. No known compression-behavior gap remains after the final matrix pass. The
   excluded items are proxy/API/dashboard/accounting surfaces rather than
   active compression behavior.
