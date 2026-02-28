"""
Pipeline generalization layer.

Provides dataclasses and a registry so new intelligence pipelines
(news, complaints, SaaS reviews, etc.) can be added with minimal
boilerplate and zero edits to infrastructure files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atlas.pipelines")


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------


@dataclass
class TaskDef:
    """Definition for a single pipeline task (maps to a scheduled_tasks row)."""

    name: str                              # e.g. "complaint_enrichment"
    module: str                            # module name under autonomous/tasks/
    handler: str = "run"
    schedule_type: str = "interval"        # "interval" or "cron"
    cron_expression: str | None = None
    interval_seconds: int | None = None    # None = resolved from config at runtime
    timeout_seconds: int = 180
    enabled: bool = True
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    interval_config_key: str | None = None  # e.g. "external_data.complaint_enrichment_interval_seconds"
    cron_config_key: str | None = None      # e.g. "external_data.complaint_analysis_cron"


@dataclass
class CleanupRule:
    """A single cleanup SQL rule executed during nightly cleanup."""

    table: str
    where_clause: str              # SQL fragment with $1 for retention_days
    retention_config_key: str      # dotted config path, e.g. "external_data.retention_days"
    result_key: str = ""           # key in cleanup result dict (auto-derived if empty)


@dataclass
class PipelineConfig:
    """Complete configuration for one intelligence pipeline."""

    name: str
    enabled_key: str               # dotted config path, e.g. "external_data.complaint_mining_enabled"
    tasks: list[TaskDef] = field(default_factory=list)
    cleanup_rules: list[CleanupRule] = field(default_factory=list)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_registry: dict[str, PipelineConfig] = {}


def register_pipeline(config: PipelineConfig) -> None:
    """Register a pipeline configuration."""
    _registry[config.name] = config
    logger.debug("Registered pipeline '%s' (%d tasks, %d cleanup rules)",
                 config.name, len(config.tasks), len(config.cleanup_rules))


def get_pipeline(name: str) -> PipelineConfig | None:
    """Get a pipeline by name."""
    return _registry.get(name)


def get_all_pipelines() -> list[PipelineConfig]:
    """Get all registered pipelines."""
    return list(_registry.values())


# ------------------------------------------------------------------
# Accessors used by infrastructure (tasks/__init__.py, scheduler.py, runner.py)
# ------------------------------------------------------------------


def get_pipeline_task_defs() -> list[tuple[str, str, str]]:
    """Return (module, handler, task_name) tuples for tasks/__init__.py registration."""
    result = []
    for pipeline in _registry.values():
        for td in pipeline.tasks:
            result.append((td.module, td.handler, td.name))
    return result


def get_pipeline_default_tasks() -> list[dict[str, Any]]:
    """Return task definition dicts for scheduler.py seeding."""
    result = []
    for pipeline in _registry.values():
        for td in pipeline.tasks:
            task_dict: dict[str, Any] = {
                "name": td.name,
                "description": td.description,
                "task_type": "builtin",
                "schedule_type": td.schedule_type,
                "timeout_seconds": td.timeout_seconds,
                "enabled": td.enabled,
                "metadata": {**td.metadata, "builtin_handler": td.name},
            }
            if td.schedule_type == "cron":
                task_dict["cron_expression"] = td.cron_expression
            else:
                task_dict["interval_seconds"] = td.interval_seconds
            result.append(task_dict)
    return result


def get_pipeline_interval_overrides() -> dict[str, str]:
    """Return {task_name: config_key} for interval-based config overrides."""
    result = {}
    for pipeline in _registry.values():
        for td in pipeline.tasks:
            if td.interval_config_key:
                result[td.name] = td.interval_config_key
    return result


def get_pipeline_cron_overrides() -> dict[str, str]:
    """Return {task_name: config_key} for cron-based config overrides."""
    result = {}
    for pipeline in _registry.values():
        for td in pipeline.tasks:
            if td.cron_config_key:
                result[td.name] = td.cron_config_key
    return result


def get_pipeline_cleanup_rules() -> list[tuple[CleanupRule, str]]:
    """Return (CleanupRule, pipeline_name) tuples for runner.py cleanup."""
    result = []
    for pipeline in _registry.values():
        for rule in pipeline.cleanup_rules:
            result.append((rule, pipeline.name))
    return result


def resolve_config_value(dotted_key: str) -> Any:
    """Navigate settings by dotted path, e.g. 'external_data.retention_days'.

    Returns the value or None if the path doesn't resolve.
    """
    from ..config import settings

    parts = dotted_key.split(".")
    obj: Any = settings
    for part in parts:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj
