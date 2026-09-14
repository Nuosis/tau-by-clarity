# CCR status redaction — September14

Captured local intention-aware reviewer requests demonstrate CCR returning deployment_http_status=503, followed by outbound PII filtering replacing503 plus the next timestamp fragment with a placeholder. The model then queried that placeholder and rejected the correct answer. Source artifacts: ~/.codex/evidence/tau-local-cache-20260914/tau-review-intention/compressed-True.json.

Null hypothesis: a status value cannot join an adjacent date during phone detection. Direct detector reproduction disproved it: `_regex_detect('deployment_http_status=503\n2026-09-14T12:02:23 INFO worker-143')` returned `('503\n2026-09', 'AMBIGUOUS_NUMERIC')`. Matching only a partial timestamp bypassed the existing date exclusion.

Phone detection now masks complete recognized datetime spans before applying its regex. Other PII recognizers remain active. This prevents partial-date matches while retaining genuine phones immediately before dates and wrapped phone numbers. The original input is not changed; masking is only the phone recognizer's search view.

26 focused detection/provider tests pass. The new log regression failed before the fix. Coverage includes503/429/200 adjacent to timestamps via LF, CRLF and space, genuine phone/email protection, wrapped/adjacent phones, round trips, and real isolated CCR retrieval through the outbound Context PII filter. These tests establish evidence preservation, not model judgment quality or cost savings. Live local reviewer rerun is recorded separately in ~/.codex/evidence/tau-efficiency-followup-20260914/.
