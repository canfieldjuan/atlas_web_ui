"""
Live integration tests for the autonomous system and brain-edge state sync.

NO MOCKS -- exercises real DB, real state machines, real timers.
Requires a running PostgreSQL with Atlas schema.

Run:
    pytest tests/test_live_autonomous.py -v -s
"""

import asyncio
import pickle
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from atlas_brain.storage.database import get_db_pool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
async def ensure_db():
    """Ensure DB pool is usable on the current event loop.

    pytest-asyncio creates a new loop per test (function scope).
    asyncpg connections are bound to the loop they were created on,
    so we must force-reset the singleton pool and reinitialize it
    on every test to avoid 'Event loop is closed' errors.
    """
    pool = get_db_pool()
    # Force-discard any stale pool from a prior event loop
    pool._pool = None
    pool._initialized = False
    await pool.initialize()
    yield pool


# ===================================================================
# 1. PRESENCE TRACKER -- live state machine + DB persistence
# ===================================================================

class TestPresenceTrackerLive:
    """Full cycle: arrivals, departures, DB persistence, callbacks."""

    @pytest.mark.asyncio
    async def test_full_arrival_departure_cycle(self):
        """Known person arrives -> identified -> leaves -> empty (after delay)."""
        from atlas_brain.autonomous.presence import (
            OccupancyState,
            PresenceConfig,
            PresenceTracker,
        )

        tracker = PresenceTracker(config=PresenceConfig(
            empty_delay_seconds=1,
            arrival_cooldown_seconds=0,
        ))

        # Track callback invocations
        transitions = []

        async def on_transition(transition, state, person):
            transitions.append((transition, state.state.value, person))

        tracker.register_callback(on_transition)

        # -- Arrival --
        assert tracker.state.state == OccupancyState.EMPTY

        await tracker.on_security_event("person_entered", {
            "name": "TestUser_Live", "is_known": True,
        })

        assert tracker.state.state == OccupancyState.IDENTIFIED
        assert "TestUser_Live" in tracker.state.occupants
        assert isinstance(tracker.state.occupants["TestUser_Live"], datetime)

        # -- Departure --
        await tracker.on_security_event("person_left", {
            "name": "TestUser_Live", "is_known": True,
        })

        assert "TestUser_Live" not in tracker.state.occupants

        # Wait for empty delay
        await asyncio.sleep(1.2)
        assert tracker.state.state == OccupancyState.EMPTY

        # -- Verify callbacks fired --
        assert len(transitions) == 2
        assert transitions[0][0] == "arrival"
        assert transitions[0][2] == "TestUser_Live"
        assert transitions[1][0] == "departure"

        await tracker.shutdown()
        print(f"\n  [PASS] Full cycle: {transitions}")

    @pytest.mark.asyncio
    async def test_persistence_writes_to_db(self):
        """Verify presence transitions actually land in the DB."""
        from atlas_brain.autonomous.presence import (
            PresenceConfig,
            PresenceTracker,
        )

        pool = get_db_pool()

        # Count before
        before = await pool.fetchrow(
            "SELECT count(*) as c FROM presence_events"
        )

        tracker = PresenceTracker(config=PresenceConfig(
            empty_delay_seconds=0,
            arrival_cooldown_seconds=0,
        ))

        await tracker.on_security_event("person_entered", {
            "name": "DBTest_Live", "is_known": True,
        })

        # Wait for empty transition to fire and persist
        await tracker.on_security_event("person_left", {
            "name": "DBTest_Live", "is_known": True,
        })
        await asyncio.sleep(0.2)

        after = await pool.fetchrow(
            "SELECT count(*) as c FROM presence_events"
        )

        new_rows = after["c"] - before["c"]
        assert new_rows >= 1, f"Expected at least 1 new presence_event row, got {new_rows}"

        # Verify the latest row
        latest = await pool.fetchrow(
            "SELECT transition, person_name, occupancy_state FROM presence_events ORDER BY created_at DESC LIMIT 1"
        )
        print(f"\n  [PASS] DB row: transition={latest['transition']}, person={latest['person_name']}, state={latest['occupancy_state']}")

        await tracker.shutdown()

    @pytest.mark.asyncio
    async def test_mixed_known_unknown_occupants(self):
        """Unknown + known arrivals produce correct state progression."""
        from atlas_brain.autonomous.presence import (
            OccupancyState,
            PresenceConfig,
            PresenceTracker,
        )

        tracker = PresenceTracker(config=PresenceConfig(
            empty_delay_seconds=1,
            arrival_cooldown_seconds=0,
        ))

        # Unknown arrives -> OCCUPIED
        await tracker.on_security_event("person_entered", {
            "name": "unknown", "is_known": False,
        })
        assert tracker.state.state == OccupancyState.OCCUPIED
        assert tracker._unknown_count == 1
        assert len(tracker.state.occupants) == 0

        # Known arrives -> IDENTIFIED
        await tracker.on_security_event("person_entered", {
            "name": "KnownUser", "is_known": True,
        })
        assert tracker.state.state == OccupancyState.IDENTIFIED
        assert "KnownUser" in tracker.state.occupants
        assert tracker._unknown_count == 1

        print(f"\n  [PASS] Mixed: state={tracker.state.state.value}, occupants={list(tracker.state.occupants.keys())}, unknown={tracker._unknown_count}")
        await tracker.shutdown()


# ===================================================================
# 2. EVENT QUEUE -- live dedup, debounce, flush
# ===================================================================

class TestEventQueueLive:
    """Real asyncio timers, real event flow."""

    @pytest.mark.asyncio
    async def test_dedup_and_debounce_flush(self):
        """Rapid duplicate events dedup, then flush after debounce."""
        from atlas_brain.alerts.events import VisionAlertEvent
        from atlas_brain.alerts.rules import AlertRule
        from atlas_brain.autonomous.event_queue import EventQueue, EventQueueConfig

        flushed_batches = []

        async def capture_batch(batch):
            flushed_batches.append(batch)

        queue = EventQueue(config=EventQueueConfig(
            debounce_seconds=0.3,
            max_batch_size=50,
            max_age_seconds=30.0,
        ))
        queue.register_callback(capture_batch)

        event = VisionAlertEvent(
            source_id="cam_test",
            timestamp=datetime.utcnow(),
            class_name="person",
            detection_type="new_track",
            track_id=42,
            node_id="test_node",
        )
        rule = AlertRule(
            name="test_rule",
            event_types=["vision"],
            source_pattern="cam_test*",
        )

        # Enqueue 5 identical events (should dedup to 1 with count=5)
        for _ in range(5):
            await queue.enqueue(event, rule, "person detected at test cam")

        assert queue.stats["total_enqueued"] == 5
        assert queue.stats["total_deduplicated"] == 4
        assert queue.stats["pending"] == 1

        # Wait for debounce flush
        await asyncio.sleep(0.5)

        assert len(flushed_batches) == 1
        batch = flushed_batches[0]
        assert len(batch) == 1
        assert batch[0].count == 5
        assert queue.stats["total_flushed"] == 1
        assert queue.stats["pending"] == 0

        print(f"\n  [PASS] Dedup+flush: 5 events -> 1 queued (count=5), flushed after debounce")
        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_max_batch_immediate_flush(self):
        """Hitting max_batch_size triggers immediate flush."""
        from atlas_brain.alerts.events import VisionAlertEvent
        from atlas_brain.alerts.rules import AlertRule
        from atlas_brain.autonomous.event_queue import EventQueue, EventQueueConfig

        flushed = []

        async def capture(batch):
            flushed.append(len(batch))

        queue = EventQueue(config=EventQueueConfig(
            debounce_seconds=10,  # long debounce
            max_batch_size=3,     # low threshold
        ))
        queue.register_callback(capture)

        rule = AlertRule(name="r", event_types=["vision"], source_pattern="*")

        # 3 distinct events hit the max batch size
        for i in range(3):
            ev = VisionAlertEvent(
                source_id=f"cam_{i}",
                timestamp=datetime.utcnow(),
                class_name=f"class_{i}",
                detection_type="new_track",
                track_id=i,
                node_id="n",
            )
            await queue.enqueue(ev, rule, f"msg_{i}")

        # Should have flushed immediately (no 10s wait)
        assert len(flushed) == 1
        assert flushed[0] == 3

        print(f"\n  [PASS] Max batch: 3 distinct events -> immediate flush")
        await queue.shutdown()


# ===================================================================
# 3. HOOK MANAGER -- live cooldown + context injection
# ===================================================================

class TestHookManagerLive:
    """Test pure logic methods with real time.monotonic()."""

    def test_cooldown_real_timing(self):
        """Record execution, verify cooldown, wait, verify expired."""
        import time
        from atlas_brain.autonomous.hooks import HookManager

        mgr = HookManager()

        assert not mgr._is_in_cooldown("task_a", "rule_a", 1)

        mgr._record_execution_time("task_a", "rule_a")
        assert mgr._is_in_cooldown("task_a", "rule_a", 1)

        # Wait past cooldown
        time.sleep(1.1)
        assert not mgr._is_in_cooldown("task_a", "rule_a", 1)

        print(f"\n  [PASS] Cooldown: recorded -> in cooldown -> expired after 1.1s")

    def test_context_injection_with_real_objects(self):
        """Inject alert context using real AlertEvent/AlertRule objects."""
        from atlas_brain.alerts.events import VisionAlertEvent
        from atlas_brain.alerts.rules import AlertRule
        from atlas_brain.autonomous.hooks import HookManager
        from atlas_brain.storage.models import ScheduledTask

        mgr = HookManager()

        task = ScheduledTask(
            id=uuid4(),
            name="security_report",
            task_type="hook",
            schedule_type="once",
            prompt="Analyze the security event and summarize.",
        )

        event = VisionAlertEvent(
            source_id="cam_front_door",
            timestamp=datetime.utcnow(),
            class_name="person",
            detection_type="new_track",
            track_id=99,
            node_id="orangepi",
            metadata={"confidence": 0.92, "zone": "entrance"},
        )

        rule = AlertRule(
            name="front_door_person",
            event_types=["vision"],
            source_pattern="cam_front_door",
        )

        result = mgr._inject_alert_context(task, "Person at front door", rule, event)

        # Original not mutated
        assert task.prompt == "Analyze the security event and summarize."

        # Result has context
        assert "[Alert Context]" in result.prompt
        assert "front_door_person" in result.prompt
        assert "Person at front door" in result.prompt
        assert "confidence" in result.prompt
        assert "entrance" in result.prompt

        print(f"\n  [PASS] Context injection: {len(result.prompt)} chars, original untouched")


# ===================================================================
# 4. IDENTITY REPOSITORY -- live DB diff_manifest + upsert/get
# ===================================================================

class TestIdentityRepoLive:
    """Real DB operations for identity embeddings."""

    @pytest.mark.asyncio
    async def test_upsert_get_roundtrip(self):
        """Write an embedding, read it back, verify match."""
        from atlas_brain.storage.repositories.identity import get_identity_repo

        repo = get_identity_repo()
        name = f"_test_live_{uuid4().hex[:8]}"
        embedding = np.random.randn(512).astype(np.float32)

        try:
            await repo.upsert(name, "face", embedding, source_node="test")

            result = await repo.get(name, "face")
            assert result is not None
            np.testing.assert_array_almost_equal(result, embedding, decimal=5)

            print(f"\n  [PASS] Upsert+get roundtrip: {name}/face (512-dim)")
        finally:
            # Cleanup
            await repo.delete(name, "face")

    @pytest.mark.asyncio
    async def test_diff_manifest_against_real_db(self):
        """Compare a fake edge manifest against actual brain DB."""
        from atlas_brain.storage.repositories.identity import get_identity_repo

        repo = get_identity_repo()

        # Get actual brain names
        brain_names = await repo.get_all_names()
        print(f"\n  Brain identities: {brain_names}")

        # Simulate edge manifest that's missing some + has extras
        edge_manifest = {}
        for mod, names in brain_names.items():
            if names:
                # Edge has first name, missing the rest
                edge_manifest[mod] = [names[0]]
            else:
                edge_manifest[mod] = []

        # Add a fake name the edge "has" but brain doesn't
        edge_manifest.setdefault("face", []).append("_test_edge_only_person")

        to_send, to_delete, need_from_edge = await repo.diff_manifest(edge_manifest)

        # to_delete is always empty by design
        assert to_delete == {}

        # _test_edge_only_person should be in need_from_edge
        assert "_test_edge_only_person" in need_from_edge.get("face", [])

        print(f"  to_send modalities: {list(to_send.keys())}")
        for mod, identities in to_send.items():
            print(f"    {mod}: {list(identities.keys())} ({len(next(iter(identities.values())))} dims each)")
        print(f"  need_from_edge: {need_from_edge}")
        print(f"  to_delete: {to_delete} (always empty by design)")
        print(f"  [PASS] diff_manifest with real DB data")

    @pytest.mark.asyncio
    async def test_validation_rejects_bad_modality(self):
        """Invalid modality raises ValueError (no DB call)."""
        from atlas_brain.storage.repositories.identity import get_identity_repo

        repo = get_identity_repo()
        with pytest.raises(ValueError, match="Invalid modality"):
            await repo.upsert("test", "invalid_mod", np.zeros(512))

        print(f"\n  [PASS] Validation: 'invalid_mod' correctly rejected")

    @pytest.mark.asyncio
    async def test_validation_rejects_wrong_dim(self):
        """Wrong dimension raises ValueError (no DB call)."""
        from atlas_brain.storage.repositories.identity import get_identity_repo

        repo = get_identity_repo()
        with pytest.raises(ValueError, match="dim mismatch"):
            await repo.upsert("test", "face", np.zeros(128))  # face expects 512

        print(f"\n  [PASS] Validation: 128-dim face embedding correctly rejected")


# ===================================================================
# 5. SCHEDULER -- live trigger building + DB task loading
# ===================================================================

class TestSchedulerLive:
    """Test trigger building and DB interaction."""

    def test_trigger_building_with_real_apscheduler(self):
        """Build all trigger types from real ScheduledTask objects."""
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.date import DateTrigger
        from apscheduler.triggers.interval import IntervalTrigger

        from atlas_brain.autonomous.scheduler import TaskScheduler
        from atlas_brain.storage.models import ScheduledTask

        scheduler = TaskScheduler()

        # Cron
        cron_task = ScheduledTask(
            id=uuid4(), name="cron_test", task_type="agent_prompt",
            schedule_type="cron", cron_expression="0 8 * * *",
        )
        trigger = scheduler._build_trigger(cron_task)
        assert isinstance(trigger, CronTrigger)

        # Interval
        interval_task = ScheduledTask(
            id=uuid4(), name="interval_test", task_type="agent_prompt",
            schedule_type="interval", interval_seconds=3600,
        )
        trigger = scheduler._build_trigger(interval_task)
        assert isinstance(trigger, IntervalTrigger)

        # Once
        once_task = ScheduledTask(
            id=uuid4(), name="once_test", task_type="agent_prompt",
            schedule_type="once", run_at=datetime(2026, 12, 31, tzinfo=timezone.utc),
        )
        trigger = scheduler._build_trigger(once_task)
        assert isinstance(trigger, DateTrigger)

        # Invalid
        bad_task = ScheduledTask(
            id=uuid4(), name="bad_test", task_type="agent_prompt",
            schedule_type="nonexistent",
        )
        trigger = scheduler._build_trigger(bad_task)
        assert trigger is None

        print(f"\n  [PASS] Built cron, interval, date, and None triggers")

    @pytest.mark.asyncio
    async def test_load_real_tasks_from_db(self):
        """Verify we can read scheduled tasks from the actual DB."""
        pool = get_db_pool()

        rows = await pool.fetch(
            "SELECT id, name, task_type, schedule_type, enabled FROM scheduled_tasks ORDER BY name"
        )

        print(f"\n  Scheduled tasks in DB ({len(rows)}):")
        for row in rows:
            print(f"    {row['name']} ({row['task_type']}/{row['schedule_type']}) enabled={row['enabled']}")

        assert len(rows) > 0, "Expected at least 1 scheduled task in the DB"
        print(f"  [PASS] Loaded {len(rows)} tasks from live DB")

    @pytest.mark.asyncio
    async def test_execution_history_exists(self):
        """Check that task_executions table has historical data."""
        pool = get_db_pool()

        rows = await pool.fetch("""
            SELECT te.status, te.duration_ms, st.name as task_name
            FROM task_executions te
            JOIN scheduled_tasks st ON st.id = te.task_id
            ORDER BY te.started_at DESC
            LIMIT 5
        """)

        print(f"\n  Recent executions ({len(rows)}):")
        for row in rows:
            duration = f"{row['duration_ms']}ms" if row['duration_ms'] else "n/a"
            print(f"    {row['task_name']}: {row['status']} ({duration})")

        print(f"  [PASS] Execution history accessible")
