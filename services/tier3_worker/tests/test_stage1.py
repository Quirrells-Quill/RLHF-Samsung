import pytest
import json
import os
from unittest.mock import AsyncMock, patch, MagicMock
from pydantic import ValidationError

from common.schemas.redis_event import RedisEventEnvelope
from common.schemas.label_studio_webhook import LSAnnotationUpdatedPayload
from common.schemas.tier3_rlhf import WiggleCacheEntry
from tier3_worker.stage1 import process_envelope

# A dummy wiggle cache entry with all required fields (including Dev 2 additions)
dummy_wiggle_cache = {
    "wiggle_seed": "seed_7781",
    "task_id": "task_a1b2c3",
    "image_id": "img_123",
    "m_initial": {"points": [[10.0, 12.0], [80.0, 12.0], [80.0, 88.0], [10.0, 88.0]]},
    "m_wiggled": {"points": [[10.1, 12.1], [80.1, 12.1], [80.1, 88.1], [10.1, 88.1]]},
    "served_at": "2026-08-18T10:15:00Z",
    "image_width": 1920,
    "image_height": 1080,
    "model_version": "v1.0",
    "is_honeypot": False,
    "m_gold": None,
    "label": "classA"
}

# A dummy payload
dummy_ls_payload = {
    "action": "ANNOTATION_UPDATED",
    "task_id": "task_a1b2c3",
    "annotation_id": "ann_9f8e7d",
    "project_id": "proj_1",
    "completed_by": "annotator_1",
    "result": [
        {
            "id": "reg1",
            "type": "polygonlabels",
            "value": {"points": [[10.5, 12.5], [80.5, 12.5], [80.5, 88.5], [10.5, 88.5]]}
        }
    ],
    "effort_telemetry": {
        "click_count": 4,
        "cursor_path_length_px": 512.7,
        "dwell_time_ms": 3400,
        "wiggle_seed": "seed_7781"
    },
    "lead_time": 14.2,
    "created_at": "2026-08-18T10:16:40Z",
    "updated_at": "2026-08-18T10:16:40Z"
}

@pytest.fixture
def mock_envelope():
    payload = LSAnnotationUpdatedPayload(**dummy_ls_payload)
    return RedisEventEnvelope(
        event_id="evt_123",
        event_type="annotation.updated",
        idempotency_key=payload.annotation_id,
        retry_count=0,
        enqueued_at="2026-08-18T10:16:40Z",
        payload=payload
    )

@pytest.fixture
def mock_wiggle_cache():
    return WiggleCacheEntry(**dummy_wiggle_cache)

@pytest.fixture
def mock_db_pool():
    pool = MagicMock()
    conn = AsyncMock()
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
    
    mock_redis.get.return_value = mock_wiggle_cache.model_dump_json()

    stage1_output = await process_envelope(mock_envelope, mock_redis, pool)

    assert stage1_output is not None
    assert stage1_output.annotation_id == "ann_9f8e7d"
    assert stage1_output.task_id == "task_a1b2c3"
    assert stage1_output.wiggle_seed == "seed_7781"
    
    assert stage1_output.effort is not None
    expected_raw_effort = (4 * 1.0) + (512.7 * 0.01) + (3400 * 0.001) # 4.0 + 5.127 + 3.4 = 12.527
    assert abs(stage1_output.effort.delta_e_raw - expected_raw_effort) < 0.001
    
    # stddev from m2/count: sqrt(2.0/2) = 1.0
    expected_norm_effort = (expected_raw_effort - 5.0) / 1.0
    assert abs(stage1_output.effort.delta_e_norm - expected_norm_effort) < 0.001
    assert stage1_output.dropped_as_bot == False
    
    assert stage1_output.model_version == "v1.0"
    assert stage1_output.image_width == 1920
    assert stage1_output.image_height == 1080

@pytest.mark.asyncio
async def test_process_envelope_duplicate(mock_envelope, mock_db_pool, mock_redis):
    pool, conn = mock_db_pool
    conn.fetchval.side_effect = [None, None] # Prior seq is None, Insert fails
    
    stage1_output = await process_envelope(mock_envelope, mock_redis, pool)
    assert stage1_output is None

@pytest.mark.asyncio
async def test_process_envelope_out_of_order(mock_envelope, mock_db_pool, mock_redis):
    pool, conn = mock_db_pool
    conn.fetchval.side_effect = [2000000000000, "ann_9f8e7d"]
    
    stage1_output = await process_envelope(mock_envelope, mock_redis, pool)
    assert stage1_output is None
