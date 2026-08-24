# Tier 3 - RLHF Reward & Rollout Contracts

Owner: Tier 3/4 team.

This module consumes `RedisEventEnvelope` (which wraps `LSAnnotationUpdatedPayload`) and produces `Stage1Output`.

## Architecture & Configuration

- `consumer.py`: Long-running async process running configurable `N` BRPOP workers pulling from `telemetry:ingest`.
- `stage1.py`: Contains `process_envelope` to extract signals and normalize effort (Dev 1).

### Sequence Number Derivation
The monotonic `sequence_no` is derived directly from `RedisEventEnvelope.enqueued_at` by converting the ISO-8601 UTC timestamp to integer milliseconds. This avoids external sequence generators and ensures that temporally later payloads for the same task have strictly higher sequence numbers.

### Bot Velocity Threshold
`EDRDE_BOT_VELOCITY_PX_MS` (default 5.0). If `cursor_path_length_px / max(dwell_time_ms, 1)` exceeds this, the payload is flagged as `dropped_as_bot=True`. 5.0 px/ms requires moving the mouse 5000 pixels per second consistently, which is beyond normal human annotation effort and strongly implies programmatic drawing or extreme noisy inputs.
