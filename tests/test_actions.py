from __future__ import annotations

import pytest
from pydantic import ValidationError

from fast_memory_write_env.actions import (
    ActionExecutionResult,
    ActionType,
    CompressMemoryAction,
    WriteMemoryAction,
    validate_memory_action,
    validate_memory_actions,
)


def test_discriminated_action_parsing() -> None:
    action = validate_memory_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-001",
            "entity_id": "account-ada",
            "content": "Account Ada prefers SMS.",
            "source_event_ids": ["event-001"],
            "fact_ids": ["fact-001"],
        }
    )

    assert isinstance(action, WriteMemoryAction)
    assert action.action_type == "write_memory"
    assert action.importance == 3


def test_invalid_action_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_memory_action({"action_type": "store_all_the_things"})


def test_action_list_validation() -> None:
    actions = validate_memory_actions(
        [
            {
                "action_type": "ignore_event",
                "event_id": "event-001",
                "reason": "Low-value dashboard view.",
            },
            {
                "action_type": "delay_index",
                "memory_id": "mem-001",
                "retry_after_ms": 500,
                "reason": "Indexing budget exhausted.",
            },
        ]
    )

    assert [action.action_type for action in actions] == ["ignore_event", "delay_index"]


def test_compress_action_rejects_target_as_source() -> None:
    with pytest.raises(ValidationError):
        CompressMemoryAction(
            source_memory_ids=["mem-001", "mem-002"],
            target_memory_id="mem-001",
            compressed_content="Compressed account summary.",
        )


def test_action_execution_result_requires_error_on_failure() -> None:
    with pytest.raises(ValidationError):
        ActionExecutionResult(
            success=False,
            action_type=ActionType.WRITE_MEMORY,
            latency_ms=1.0,
            storage_tokens_delta=0,
        )

    result = ActionExecutionResult(
        success=False,
        action_type="write_memory",
        latency_ms=1.0,
        storage_tokens_delta=0,
        error="memory_id already exists",
    )

    assert result.action_type == ActionType.WRITE_MEMORY
