import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from pydantic import ValidationError
import json

from common.schemas.redis_event import RedisEventEnvelope
from tier3_worker.consumer import worker

@pytest.mark.asyncio
async def test_consumer_drops_invalid_json():
    redis_pool = AsyncMock()
    # First return a bad payload, then block forever (simulated by CancelledError)
    redis_pool.brpop.side_effect = [
        ("telemetry:ingest", b"not a json"),
        asyncio.CancelledError()
    ]
    
    db_pool = AsyncMock()
    
    with pytest.raises(asyncio.CancelledError):
        await worker(0, redis_pool, db_pool)
        
    assert redis_pool.brpop.call_count == 2

@pytest.mark.asyncio
async def test_consumer_drops_validation_error():
    redis_pool = AsyncMock()
    
    # Valid JSON but invalid schema
    bad_schema = json.dumps({"event_id": "123"}).encode('utf-8')
    redis_pool.brpop.side_effect = [
        ("telemetry:ingest", bad_schema),
        asyncio.CancelledError()
    ]
    
    db_pool = AsyncMock()
    
    with pytest.raises(asyncio.CancelledError):
        await worker(0, redis_pool, db_pool)
        
    assert redis_pool.brpop.call_count == 2
