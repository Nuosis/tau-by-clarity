# OpenAI subscription transport repair

The model registry selected subscription transport only for model IDs in its
bundled Codex catalog. Configured Luna, Sol, and Astra IDs were absent, so they
fell through to synthetic paid-API definitions even with OAuth credentials.

Subscription credentials now select Codex transport independently of that
catalog. Requested model IDs and router reasoning settings are preserved.
Unsupported model requests remain subscription requests and surface rejection;
there is no paid API retry. Expired/unusable OpenAI OAuth in the synchronous
resolver raises rather than falling back to a stored or environment API key.
A models.json key no longer overrides the selected subscription credential.
API-key-only configurations retain API transport.

Validation: five focused regression tests pass against source and the installed
/Users/marcusswift/.venv runtime. Tests capture outgoing HTTP through the normal
stream dispatcher, inject a 403 at the network boundary, and confirm that the
request uses the subscription endpoint and credential without retrying elsewhere.
The baseline exposed catalog selection and expired-token fallback failures; an
additional valid custom-provider configuration exposed key precedence.

Live verification through installed runtime and saved OAuth: gpt-5.6-luna,
gpt-5.6-sol, and gpt-6-astra each completed one tiny subscription-endpoint request,
20 input + 5 output tokens each (75 total). All four real configured router tiers
resolve to https://chatgpt.com/backend-api with openai-codex-responses; original
model IDs and reasoning levels are unchanged. This verifies model access and
transport, not long-running task quality or final billing reconciliation.

Restart existing Tau clients to load the changes. Price-table/cache-write
accounting is a separate outstanding issue; this repair does not claim to fix it.
