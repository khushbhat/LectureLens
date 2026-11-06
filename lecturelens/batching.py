"""Stage 8 — Quota-aware batching.

Consecutive segments are packed into batches under the active provider's
per-call input budget (minus the prompt template). Packing is balanced
(n = ceil(total / budget) batches of ~equal size) so a 60-min lecture
becomes a handful of map calls — never one call per slide.
"""

from __future__ import annotations

import math

from .utils import estimate_tokens


def pack_batches(seg_texts: list[tuple[int, str]], budget_tokens: int) -> list[dict]:
    """seg_texts: [(segment_id, rendered_text)] in order -> [{batch_id, segment_ids, est_tokens}]."""
    sizes = [(sid, estimate_tokens(t) + 2) for sid, t in seg_texts]
    total = sum(s for _, s in sizes)
    if not sizes:
        return []
    budget = max(200, budget_tokens)
    n = max(1, math.ceil(total / budget))
    target = min(budget, math.ceil(total / n) * 1.05)
    batches, cur, cur_tok = [], [], 0
    for sid, tok in sizes:
        if cur and (cur_tok + tok > budget or cur_tok + tok > target * 1.15 and cur_tok >= target * 0.85):
            batches.append((cur, cur_tok))
            cur, cur_tok = [], 0
        cur.append(sid)
        cur_tok += tok
    if cur:
        batches.append((cur, cur_tok))
    return [{"batch_id": i, "segment_ids": ids, "est_tokens": tok} for i, (ids, tok) in enumerate(batches)]
