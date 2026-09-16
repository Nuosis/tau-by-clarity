"""Small, reversible review evidence with stable material before changing targets."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, is_dataclass

from ..active_compression import _ccr
from ..active_compression.search import search_original

logger = logging.getLogger(__name__)

LONG_EVIDENCE_CHARS = 4096
FOCUSED_EVIDENCE_CHARS = 2048
MAX_FOCUSED_EVIDENCE_ITEMS = 6
MAX_FOCUSED_EVIDENCE_CHARS = 8192
REVIEW_CONTINUATION_PREFIX = 'Completion review requires further work within the existing user authorization:'
_HANDLE = re.compile(r'\[CCR:([0-9a-fA-F]{12})\]')


def _content_text(message):
    content = message.get('content', '')
    if isinstance(content, str):
        return content
    return ' '.join(block.get('text', '') for block in content if isinstance(block, dict))


def _review_start(originals):
    for index in range(len(originals) - 1, -1, -1):
        message = originals[index]
        if message.get('role') == 'user' and _content_text(message).startswith(REVIEW_CONTINUATION_PREFIX):
            return index
    return 0


def build_review_payload(context, intention=None, *, task_start=0):
    originals = [m.model_dump(mode='json') if hasattr(m, 'model_dump') else asdict(m) if is_dataclass(m) else dict(m)
                 for m in context.messages]
    query_parts = []
    if intention:
        query_parts.extend([intention.outcome, *intention.completion_evidence])
    for message in originals[-4:]:
        if message.get('role') in {'user', 'assistant'}:
            query_parts.append(_content_text(message))
    query = ' '.join(dict.fromkeys(part.strip() for part in query_parts if part.strip()))[:2048]
    review_start = _review_start(originals)
    task_start = min(max(0, int(task_start or 0)), len(originals))
    start = max(review_start, task_start)
    messages, focused = [], []
    handle_candidates = []
    store = None
    seen_handles = set()
    focused_chars = 0

    def prefetch(handle, ref):
        nonlocal store, focused_chars
        handle = handle.lower()
        if (handle in seen_handles or not query or
                len(focused) >= MAX_FOCUSED_EVIDENCE_ITEMS or
                focused_chars >= MAX_FOCUSED_EVIDENCE_CHARS):
            return
        seen_handles.add(handle)
        store = store or _ccr()
        source = store.get(handle)
        if source is None:
            return
        result = search_original(source, query, max_items=8)
        if result.get('kept_items', 0) <= 0:
            return
        excerpt = result.get('text', '')[:min(FOCUSED_EVIDENCE_CHARS,
                                                MAX_FOCUSED_EVIDENCE_CHARS - focused_chars)]
        if excerpt:
            focused.append({'ref': ref, 'handle': handle, 'excerpt': excerpt, 'partial': True})
            focused_chars += len(excerpt)

    for index, original in enumerate(originals):
        value = {k: v for k, v in original.items()
                 if k not in {'usage', 'api', 'provider', 'model', 'timestamp'}}
        # These are transport annotations, not task evidence. Do not recurse
        # into tool arguments: business null/false values must remain intact.
        if value.get('error_message') is None:
            value.pop('error_message', None)
        if isinstance(value.get('content'), list):
            clean_blocks = []
            for block in value['content']:
                if not isinstance(block, dict):
                    clean_blocks.append(block)
                    continue
                block = dict(block)
                for key in ('text_signature', 'cache_control', 'cache_zone', 'mutable',
                            'arguments_raw', 'arguments_repaired_raw',
                            'arguments_parse_error', 'thought_signature'):
                    if block.get(key) is None:
                        block.pop(key, None)
                if block.get('arguments_repair_applied') is False:
                    block.pop('arguments_repair_applied')
                clean_blocks.append(block)
            value['content'] = clean_blocks
        if value.get('role') == 'toolResult':
            value.pop('details', None)
            blocks = []
            for block in value.get('content', []):
                block = dict(block)
                text = block.get('text', '')
                handle_match = _HANDLE.search(text)
                if block.get('type') == 'text' and len(text) > LONG_EVIDENCE_CHARS and not handle_match:
                    try:
                        store = store or _ccr()
                        source = text
                        handle = store.put(source, tool_name=value.get('tool_name'),
                                           compression_strategy='review_evidence')
                        # Query-independent preview remains stable across reviews.
                        block['text'] = (f'[CCR:{handle}] Abbreviated review evidence; '
                                         'the host retained omitted details for focused review evidence.\n'
                                         + text[:256] + '\n...\n' + text[-256:])
                        handle_candidates.append((index, handle, f'message:{index}'))
                    except Exception as exc:
                        # Compression/search is an optimization. If storage or
                        # retrieval fails, preserve the supplied evidence.
                        block = dict(block)
                        block['text'] = text
                        logger.warning('Review evidence projection unavailable: %s', type(exc).__name__)
                blocks.append(block)
            value['content'] = blocks
        if value.get('role') == 'toolResult':
            for match in _HANDLE.finditer(_content_text(value)):
                handle_candidates.append((index, match.group(1), f'message:{index}'))
        if index >= start:
            messages.append({'ref': f'message:{index}', 'message': value})
    # Preserve the bounded budget for the correction pass first, then use the
    # nearest earlier source evidence from this user request.  Delta review can
    # validate against prior evidence without retransmitting the prior messages.
    ordered_candidates = (
        [item for item in handle_candidates if item[0] >= start]
        + list(reversed([item for item in handle_candidates
                         if task_start <= item[0] < start]))
    )
    for _, handle, ref in ordered_candidates:
        try:
            prefetch(handle, ref)
        except Exception as exc:
            logger.warning('Review evidence prefetch unavailable: %s', type(exc).__name__)
    # Growing history and changing intention/query come after stable instructions
    # and capabilities. Never rewrite the worker's context or stored transcript.
    payload = json.dumps({
        'working_instructions': context.system_prompt,
        'worker_capabilities': [{'name': t.name, 'description': t.description}
                                for t in sorted(context.tools, key=lambda t: t.name)],
        'intention': intention.model_dump() if intention else None,
        'review_scope': {'mode': 'delta' if review_start else 'full',
                         'starts_at': f'message:{start}'},
        'messages': messages,
        'focused_evidence': focused,
        'candidate': f'message:{len(originals)-1}',
    }, default=str, ensure_ascii=False, separators=(',', ':'))
    return payload, {'focused_evidence_count': len(focused),
                     'focused_evidence_characters': focused_chars,
                     'message_count': len(messages), 'review_scope': 'delta' if review_start else 'full'}
