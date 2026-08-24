import pytest
import json
import os
from unittest.mock import AsyncMock, patch, MagicMock

from common.schemas.redis_event import RedisEventEnvelope
from common.schemas.label_studio_webhook import LSAnnotationUpdatedPayload
from common.schemas.tier3_rlhf import WiggleCacheEntry
from services.tier3_worker.stage1 import process_envelope

@pytest.fixture
def mock_envelope():
    mock_path = os.path.join(os.path.dirname(__file__), "mocks", "ls_webhook_payload.json")
    with open(mock_path, "r") as f:
        ls_payload_data = json.load(f)
    
    payload = LSAnnotationUpdatedPayload(**ls_payload_data)
    envelope = RedisEventEnvelope(
        event_id="evt_123",
        event_type="annotation.updated",
        idempotency_key=payload.annotation_id,
        retry_count=0,
        enqueued_at="2026-08-18T10:16:40Z",
        payload=payload
    )
    return envelope

@pytest.fixture
def mock_wiggle_cache():
    mock_path = os.path.join(os.path.dirname(__file__), "mocks", "wiggle_cache_entry.json")
    with open(mock_path, "r") as f:
        data = json.load(f)
    return WiggleCacheEntry(**data)

@pytest.fixture
def mock_db_pool():
    pool = MagicMock()
    conn = AsyncMock()
    # Setup acquire context manager
    pool.acquire.return_value.__aenter__.return_value = conn
    return pool, conn

@pytest.fixture
def mock_redis():
    redis_pool = AsyncMock()
    return redis_pool

@pytest.mark.asyncio
async def test_process_envelope_success(mock_envelope, mock_wiggle_cache, mock_db_pool, mock_redis):
    pool, conn = mock_db_pool
    
    # 1. Mock out-of-order check (return None for prior sequence)
    # 2. Mock INSERT (return annotation_id indicating success)
    conn.fetchval.side_effect = [None, "ann_9f8e7d"]
    
    # 3. Mock Welford update (return updated stats)
    conn.fetchrow.return_value = {"count": 2, "mean": 5.0, "m2": 2.0}
    
    # Mock Redis get
    mock_redis.get.return_value = mock_wiggle_cache.model_dump_json()

    stage1_output = await process_envelope(mock_envelope, mock_redis, pool)

    assert stage1_output is not None
    assert stage1_output.annotation_id == "ann_9f8e7d"
    assert stage1_output.task_id == "task_a1b2c3"
    assert stage1_output.wiggle_seed == "seed_7781"
    
    assert stage1_output.effort is not None
    # 4 clicks * 1.0 + 512.7 * 0.01 + 3400 * 0.001 = 4.0 + 5.127 + 3.4 = 12.527
    expected_raw_effort = 12.527
    assert abs(stage1_output.effort.delta_e_raw - expected_raw_effort) < 0.001
    
    # stddev from m2/count: sqrt(2.0/2) = 1.0
    expected_norm_effort = (expected_raw_effort - 5.0) / 1.0
    assert abs(stage1_output.effort.delta_e_norm - expected_norm_effort) < 0.001
    assert stage1_output.dropped_as_bot == False

@pytest.mark.asyncio
async def test_process_envelope_duplicate(mock_envelope, mock_db_pool, mock_redis):
    pool, conn = mock_db_pool
    
    # Mock prior_seq as None, then INSERT returns None (duplicate)
    conn.fetchval.side_effect = [None, None]
    
    stage1_output = await process_envelope(mock_envelope, mock_redis, pool)
    assert stage1_output is None

@pytest.mark.asyncio
async def test_process_envelope_out_of_order(mock_envelope, mock_db_pool, mock_redis):
    pool, conn = mock_db_pool
    
    # Mock prior_seq > sequence_no (e.g. sequence_no for 2026-08-18 is ~1787048200000, so we return larger)
    conn.fetchval.side_effect = [2000000000000, "ann_9f8e7d"]
    
    stage1_output = await process_envelope(mock_envelope, mock_redis, pool)
    assert stage1_output is None

@pytest.mark.asyncio
async def test_process_envelope_bot_detected(mock_envelope, mock_wiggle_cache, mock_db_pool, mock_redis):
    pool, conn = mock_db_pool
    conn.fetchval.side_effect = [None, "ann_9f8e7d"]
    conn.fetchrow.return_value = {"count": 2, "mean": 5.0, "m2": 2.0}
    mock_redis.get.return_value = mock_wiggle_cache.model_dump_json()

    # Modify envelope to have insane path length to trigger bot velocity > 5.0
    mock_envelope.payload.effort_telemetry.cursor_path_length_px = 50000.0
    mock_envelope.payload.effort_telemetry.dwell_time_ms = 1000

    stage1_output = await process_envelope(mock_envelope, mock_redis, pool)
    
    assert stage1_output is not None
    assert stage1_output.dropped_as_bot == True
