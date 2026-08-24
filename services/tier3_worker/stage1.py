import os
import logging
from typing import Optional
from datetime import datetime, timezone
import asyncpg
import redis.asyncio as redis

from common.schemas.redis_event import RedisEventEnvelope
from common.schemas.tier3_rlhf import (
    Stage1Output, WiggleCacheEntry, NormalizedEffortScore, EffortWeights
)

logger = logging.getLogger("tier3_stage1")

def derive_sequence_no(enqueued_at: str) -> int:
    # Derive monotonic sequence_no from enqueued_at.
    # ISO-8601 UTC string -> timestamp in milliseconds.
    try:
        dt = datetime.fromisoformat(enqueued_at.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except ValueError:
        return int(datetime.now(timezone.utc).timestamp() * 1000)

async def process_envelope(envelope: RedisEventEnvelope, redis_pool: redis.Redis, db_pool: asyncpg.Pool) -> Optional[Stage1Output]:
    payload = envelope.payload
    sequence_no = derive_sequence_no(envelope.enqueued_at)
    
    # 3. Sequence check via Postgres
    # tier3.processed_annotations: annotation_id, task_id, event_id, sequence_no, processed_at, accepted, reject_reason
    async with db_pool.acquire() as conn:
        # Check for out-of-order
        prior_seq = await conn.fetchval(
            "SELECT MAX(sequence_no) FROM tier3.processed_annotations WHERE task_id = $1", 
            payload.task_id
        )
        if prior_seq is not None and sequence_no < prior_seq:
            logger.warning(f"Out of order sequence for task {payload.task_id} (seq {sequence_no} < prior {prior_seq})")
            return None
            
        # Race-safe INSERT ON CONFLICT
        insert_query = """
            INSERT INTO tier3.processed_annotations 
            (annotation_id, task_id, event_id, sequence_no, accepted, reject_reason)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (annotation_id) DO NOTHING
            RETURNING annotation_id
        """
        inserted = await conn.fetchval(
            insert_query, 
            payload.annotation_id, payload.task_id, envelope.event_id, sequence_no, True, None
        )
        if not inserted:
            logger.warning(f"Duplicate annotation_id {payload.annotation_id}")
            return None

    # 4. Fetch WiggleCacheEntry from Redis
    wiggle_seed = payload.effort_telemetry.wiggle_seed
    if not wiggle_seed:
        return None

    raw_cache = await redis_pool.get(f"wiggle_cache:{wiggle_seed}")
    if not raw_cache:
        logger.error(f"Missing WiggleCacheEntry for task_id={payload.task_id} annotation_id={payload.annotation_id} wiggle_seed={wiggle_seed}")
        return None
        
    try:
        cache_entry = WiggleCacheEntry.model_validate_json(raw_cache)
    except Exception as e:
        logger.error(f"Failed to parse WiggleCacheEntry for task_id={payload.task_id} wiggle_seed={wiggle_seed}: {e}")
        return None

    # 5. Extract BiometricSignals
    click_count = payload.effort_telemetry.click_count
    cursor_path = payload.effort_telemetry.cursor_path_length_px
    dwell_time = payload.effort_telemetry.dwell_time_ms

    # 6. Biometric Effort Engine
    w1 = float(os.getenv("EDRDE_W1_CLICKS", "1.0"))
    w2 = float(os.getenv("EDRDE_W2_PATH", "0.01"))
    w3 = float(os.getenv("EDRDE_W3_DWELL", "0.001"))
    
    delta_e_raw = (w1 * click_count) + (w2 * cursor_path) + (w3 * dwell_time)

    # 7. Z-score normalization & Sanity Filter
    # Welford's online algorithm in Postgres
    async with db_pool.acquire() as conn:
        # Atomic UPDATE of running stats
        welford_query = """
            UPDATE tier3.effort_population_stats
            SET count = count + 1,
                mean = mean + ($1 - mean) / (count + 1),
                m2 = m2 + ($1 - mean) * ($1 - (mean + ($1 - mean) / (count + 1))),
                updated_at = now()
            WHERE id = 1
            RETURNING count, mean, m2
        """
        stats = await conn.fetchrow(welford_query, delta_e_raw)
        
        if stats and stats['count'] > 1:
            mean = stats['mean']
            variance = stats['m2'] / stats['count']
            stddev = variance ** 0.5
        else:
            # First row or failed
            mean = delta_e_raw
            stddev = 1.0

    epsilon = 1e-5
    safe_stddev = max(stddev, epsilon)
    delta_e_norm = (delta_e_raw - mean) / safe_stddev

    # Bot velocity threshold
    # cursor_path_length_px / max(dwell_time_ms, 1) > threshold
    bot_threshold = float(os.getenv("EDRDE_BOT_VELOCITY_PX_MS", "5.0"))
    safe_dwell = max(dwell_time, 1)
    velocity = cursor_path / safe_dwell
    dropped_as_bot = (velocity > bot_threshold)

    # 8. Handoff in-process -> construct Stage1Output
    effort = NormalizedEffortScore(
        delta_e_raw=delta_e_raw,
        delta_e_norm=delta_e_norm,
        dropped_as_bot=dropped_as_bot,
        population_mean=mean,
        population_stddev=stddev
    )

    ls_result_dicts = [r.model_dump() for r in payload.result]

    stage1 = Stage1Output(
        annotation_id=payload.annotation_id,
        task_id=payload.task_id,
        wiggle_seed=wiggle_seed,
        m_initial=cache_entry.m_initial,
        m_wiggled=cache_entry.m_wiggled,
        ls_result=ls_result_dicts,
        effort=effort,
        dropped_as_bot=dropped_as_bot,
        model_version=getattr(cache_entry, "model_version", None)
    )

    return stage1
