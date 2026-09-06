import os
import sys
import asyncio
import logging
import asyncpg
import redis.asyncio as redis
from pydantic import ValidationError

from common.schemas.redis_event import RedisEventEnvelope
from services.tier3_worker.stage1 import process_envelope
from services.tier3_worker.stage2 import process_stage1, configure, shutdown

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tier3_consumer")

async def worker(worker_id: int, redis_pool: redis.Redis, db_pool: asyncpg.Pool):
    logger.info(f"Worker {worker_id} started")
    while True:
        try:
            payload_bytes = None
            # 1. BRPOP telemetry:ingest
            result = await redis_pool.brpop("telemetry:ingest", timeout=0)
            if not result:
                continue
            
            queue_name, payload_bytes = result
            
            # 2. Deserialize as RedisEventEnvelope
            try:
                envelope = RedisEventEnvelope.model_validate_json(payload_bytes)
            except ValidationError as e:
                logger.error(f"Worker {worker_id} dropped invalid payload: {e}")
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} failed to parse payload: {e}")
                continue
                
            # 3. Process Envelope (Dev 1 stage 1)
            stage1_output = await process_envelope(envelope, redis_pool, db_pool)
            
            # 4. Handoff to Dev 2 (Stage 2)
            if stage1_output:
                logger.info(f"Worker {worker_id} generated Stage1Output for {stage1_output.task_id}, handing off to Dev 2")
                await process_stage1(stage1_output)
                
        except Exception as e:
            logger.error(f"Worker {worker_id} encountered unhandled exception: {e}")
            try:
                # Park the message back on the queue to avoid silent data loss
                if payload_bytes:
                    await redis_pool.lpush("telemetry:ingest", payload_bytes)
            except Exception as push_err:
                logger.error(f"Worker {worker_id} failed to push back payload: {push_err}")
            await asyncio.sleep(1)

async def main():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rlhf")
    concurrency = int(os.getenv("TIER3_CONCURRENCY", "4"))
    
    redis_pool = redis.Redis.from_url(redis_url, decode_responses=True)
    
    try:
        db_pool = await asyncpg.create_pool(dsn=db_url)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
        
    configure()
        
    tasks = []
    for i in range(concurrency):
        tasks.append(asyncio.create_task(worker(i, redis_pool, db_pool)))
        
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await shutdown()

if __name__ == "__main__":
    asyncio.run(main())
