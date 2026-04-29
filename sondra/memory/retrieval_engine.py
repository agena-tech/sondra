from __future__ import annotations

import contextlib
import logging
from typing import Any


logger = logging.getLogger(__name__)


def search_ranked_hits(
    store: Any,
    hit_cls: type,
    query: str,
    top_k: int = 8,
    candidate_limit: int = 500,
    session_id: str | None = None,
) -> list[Any]:
    raw_query = str(query or "").strip()
    q = store._canonicalize_query(raw_query)
    if not q:
        return []

    top_k = max(1, min(top_k, 20))
    candidate_limit = max(top_k * 4, min(max(1, int(candidate_limit)), 3000))

    query_tokens = store._tokenize(q)
    dedup_query_tokens: list[str] = []
    seen_query_tokens: set[str] = set()
    for token in query_tokens:
        if token in seen_query_tokens:
            continue
        seen_query_tokens.add(token)
        dedup_query_tokens.append(token)
    query_tokens = dedup_query_tokens

    query_embedding = store._embed_text(q)
    store._vector_available = bool(query_embedding)
    if store.hybrid_enabled and not store._vector_available:
        logger.warning("[MEMORY] vector disabled for this query")

    session_value = str(session_id or "").strip()

    rows: list[tuple] = []
    if raw_query:
        with contextlib.suppress(Exception):
            rows.extend(store._search_fts_rows(f'"{raw_query}"', candidate_limit, session_id=session_value))
    with contextlib.suppress(Exception):
        rows.extend(store._search_fts_rows(f'"{q}"', candidate_limit, session_id=session_value))
    with contextlib.suppress(Exception):
        rows.extend(store._search_fts_rows(q, candidate_limit, session_id=session_value))

    for token in query_tokens[:6]:
        with contextlib.suppress(Exception):
            rows.extend(store._search_fts_rows(f'"{token}"', candidate_limit, session_id=session_value))
        if len(rows) >= candidate_limit:
            break

    dedup_rows: list[tuple] = []
    seen_row_ids: set[int] = set()
    for row in rows:
        row_id = int(row[0])
        if row_id in seen_row_ids:
            continue
        seen_row_ids.add(row_id)
        dedup_rows.append(row)
        if len(dedup_rows) >= candidate_limit:
            break
    rows = dedup_rows

    if not rows:
        recent = store.recent_conversation(limit=max(top_k * 2, top_k), session_id=session_value)
        fallback_hits: list[Any] = []
        for msg in recent:
            raw_content = str(msg.content or "")
            if store._is_noisy_memory_body(raw_content):
                continue
            role = str(msg.role or "").strip().lower()
            _, body = store._extract_role_and_body(raw_content)
            lexical = store._lexical_score(query_tokens, body)
            recency = store._recency_score(str(msg.timestamp or ""))
            temporal = store._temporal_decay_score(str(msg.timestamp or ""))
            recency_effective = recency * temporal
            final = store._combine_scores(
                lexical=lexical,
                fts_score=None,
                vector_score=None,
                recency=recency_effective,
                importance=0.5,
                created_at=str(msg.timestamp or ""),
            )
            fallback_hits.append(
                hit_cls(
                    id=int(msg.id),
                    timestamp=str(msg.timestamp or ""),
                    role=role,
                    content=raw_content,
                    body=body,
                    lexical_score=lexical,
                    recency_score=recency,
                    fts_score=0.0,
                    vector_score=0.0,
                    temporal_score=temporal,
                    final_score=final,
                )
            )
        return sorted(fallback_hits, key=lambda h: h.final_score, reverse=True)[:top_k]

    embeddings: dict[int, list[float]] = {}
    if store.hybrid_enabled and store._vector_available:
        vector_candidate_limit = max(24, min(len(rows), max(top_k * 8, 64)))
        vector_rows = rows[:vector_candidate_limit]
        embeddings = store._load_embeddings_for_rows(
            vector_rows,
            compute_missing=False,
        )
    hits_by_id: dict[int, Any] = {}

    for row in rows:
        row_id = int(row[0])
        timestamp = str(row[1])
        role = str(row[2]).lower()
        raw_content = str(row[3])
        fts_rank = row[4] if len(row) > 4 else 0.0

        if store._is_noisy_memory_body(raw_content):
            continue

        _, body = store._extract_role_and_body(raw_content)
        lexical = store._lexical_score(query_tokens, body)
        recency = store._recency_score(timestamp)
        fts_score = store._fts_rank_to_score(fts_rank)
        temporal = store._temporal_decay_score(timestamp)
        recency_effective = recency * temporal

        vector_score = 0.0
        if store.hybrid_enabled and store._vector_available:
            vec = embeddings.get(row_id, [])
            if vec:
                vector_score = store._cosine_similarity(query_embedding, vec)
        use_vector = vector_score if store.hybrid_enabled and store._vector_available and vector_score > 0 else None
        use_fts = fts_score if fts_score > 0 else None
        final = store._combine_scores(
            lexical=lexical,
            fts_score=use_fts,
            vector_score=use_vector,
            recency=recency_effective,
            importance=0.5,
            created_at=timestamp,
        )

        hit = hit_cls(
            id=row_id,
            timestamp=timestamp,
            role=role,
            content=raw_content,
            body=body,
            lexical_score=lexical,
            recency_score=recency,
            fts_score=fts_score,
            vector_score=vector_score,
            temporal_score=temporal,
            final_score=final,
        )
        existing = hits_by_id.get(row_id)
        if not existing or hit.final_score > existing.final_score:
            hits_by_id[row_id] = hit

    ranked = sorted(hits_by_id.values(), key=lambda h: h.final_score, reverse=True)
    selected = store._apply_mmr_selection(ranked, embeddings, top_k)
    selected_rows = selected[:top_k]
    with contextlib.suppress(Exception):
        store._reinforce_selected_rows(selected_rows)
    return selected_rows


def search_conversation(
    store: Any,
    hit_cls: type,
    query: str,
    top_k: int = 8,
    session_id: str | None = None,
) -> list[str]:
    q = str(query or "").strip()
    if not q:
        return []
    top_k = max(1, min(top_k, 20))
    cached = store._read_query_cache(q, top_k, session_id=session_id)
    if cached is not None:
        return cached
    hits = search_ranked_hits(
        store,
        hit_cls,
        q,
        top_k=top_k,
        candidate_limit=top_k * 20,
        session_id=session_id,
    )
    values = [str(hit.content) for hit in hits]
    values = store._dedupe_memory_lines(values)
    result = store._apply_content_char_budget(values)[:top_k]
    if result:
        store._write_query_cache(q, top_k, result, session_id=session_id)
    return result


def search_conversation_structured(
    store: Any,
    hit_cls: type,
    query: str,
    top_k: int = 8,
    candidate_limit: int = 500,
    session_id: str | None = None,
) -> list[dict[str, str | float | int]]:
    query_text = str(query or "")
    safe_top_k = max(1, min(int(top_k), 20))
    safe_candidate_limit = max(1, int(candidate_limit))
    cached = store._read_structured_query_cache(
        query=query_text,
        top_k=safe_top_k,
        candidate_limit=safe_candidate_limit,
        session_id=session_id,
    )
    if cached is not None:
        return cached
    hits = search_ranked_hits(
        store,
        hit_cls,
        query=query_text,
        top_k=safe_top_k,
        candidate_limit=safe_candidate_limit,
        session_id=session_id,
    )
    rows = [store._to_structured_hit(hit, idx) for idx, hit in enumerate(hits, start=1)]
    rows = [row for row in rows if str(row.get("role", "")).strip().lower() == "user"]
    rows = store._dedupe_structured_rows(rows, safe_top_k)
    if rows:
        store._write_structured_query_cache(
            query=query_text,
            top_k=safe_top_k,
            candidate_limit=safe_candidate_limit,
            rows=rows,
            session_id=session_id,
        )
    return rows
