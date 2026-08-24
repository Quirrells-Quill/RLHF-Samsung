import asyncio
import json
import os
import redis.asyncio as redis

async def main():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_pool = redis.Redis.from_url(redis_url, decode_responses=True)
    
    # 1. Load fixtures
    with open("tests/mocks/ls_webhook_payload.json", "r") as f:
        ls_payload_data = json.load(f)
        
    with open("tests/mocks/wiggle_cache_entry.json", "r") as f:
        wiggle_cache_data = json.load(f)
        
    wiggle_seed = wiggle_cache_data["wiggle_seed"]
    annotation_id = ls_payload_data["annotation_id"]
    
    envelope = {
        "event_id": "evt_test_123",
        "event_type": "annotation.updated",
        "idempotency_key": annotation_id,
        "retry_count": 0,
        "enqueued_at": "2026-08-18T10:16:40Z",
        "payload": ls_payload_data
    }
    
    # 2. Push to Redis
    print(f"Setting wiggle_cache:{wiggle_seed} ...")
    await redis_pool.set(f"wiggle_cache:{wiggle_seed}", json.dumps(wiggle_cache_data), ex=86400)
    
    print(f"Pushing envelope to telemetry:ingest ...")
    await redis_pool.lpush("telemetry:ingest", json.dumps(envelope))
    
    print("Done! You can now run consumer.py to test the pipeline.")

if __name__ == "__main__":
    asyncio.run(main())
