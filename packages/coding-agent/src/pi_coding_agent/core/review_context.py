"""Small, reversible review evidence with stable material before changing targets."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
import re

from ..active_compression import _ccr
from ..active_compression.search import search_original

logger = logging.getLogger(__name__)

LONG_EVIDENCE_CHARS = 4096
FOCUSED_EVIDENCE_CHARS = 2048
_HANDLE = re.compile(r'\[CCR:([0-9a-fA-F]{12})\]')


def build_review_payload(context, intention=None):
    originals = [m.model_dump(mode='json') if hasattr(m, 'model_dump') else asdict(m) if is_dataclass(m) else dict(m)
                 for m in context.messages]
    query = intention.outcome if intention else ''
    if not query:
        for message in reversed(originals):
            if message.get('role') == 'user':
                content = message.get('content', '')
                query = content if isinstance(content, str) else ' '.join(
                    b.get('text', '') for b in content if isinstance(b, dict))
                break
    query = query[:1024]
    messages, focused = [], []
    store = None
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
                                         'use ccr_retrieve with a focused query for omitted details.\n'
                                         + text[:256] + '\n...\n' + text[-256:])
                        if source is not None and query:
                            result = search_original(source, query, max_items=8)
                            excerpt = result.get('text', '')[:FOCUSED_EVIDENCE_CHARS]
                            if excerpt:
                                focused.append({'ref': f'message:{index}', 'handle': handle,
                                                'excerpt': excerpt,
                                                'partial': True})
                    except Exception as exc:
                        # Compression/search is an optimization. If storage or
                        # retrieval fails, preserve the supplied evidence.
                        block = dict(block)
                        block['text'] = text
                        logger.warning('Review evidence projection unavailable: %s', type(exc).__name__)

                blocks.append(block)
            value['content'] = blocks
        messages.append({'ref': f'message:{index}', 'message': value})
    # Growing history and changing intention/query come after stable instructions
    # and capabilities. Never rewrite the worker's context or stored transcript.
    payload = json.dumps({
        'working_instructions': context.system_prompt,
        'worker_capabilities': [{'name': t.name, 'description': t.description}
                                for t in sorted(context.tools, key=lambda t: t.name)],
        'messages': messages,
        'intention': intention.model_dump() if intention else None,
        'focused_evidence': focused,
        'candidate': f'message:{len(messages)-1}',
    }, default=str, ensure_ascii=False, separators=(',', ':'))
    return payload, {'focused_evidence_count': len(focused)}
