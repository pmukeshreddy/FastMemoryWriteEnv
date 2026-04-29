"""Tests for the policy-proposal layer and plan validator/compiler."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fast_memory_write_env.actions import (
    CompressMemoryAction,
    CompressMemoryProposal,
    DelayIndexAction,
    IgnoreEventAction,
    IndexNowAction,
    MarkStaleAction,
    PolicyPlanError,
    UpdateMemoryAction,
    WriteMemoryAction,
    WriteMemoryProposal,
    compile_policy_actions,
    deterministic_memory_id,
    validate_action_plan,
    validate_policy_action,
    validate_policy_actions,
)


def _write_proposal(content: str = "alpha", source_event_id: str = "event-1") -> WriteMemoryProposal:
    return WriteMemoryProposal(
        entity_id="account-ada",
        content=content,
        source_event_ids=[source_event_id],
    )


def test_write_memory_proposal_rejects_llm_supplied_memory_id() -> None:
    with pytest.raises(ValidationError):
        validate_policy_action(
            {
                "action_type": "write_memory",
                "memory_id": "mem-llm-authored",
                "entity_id": "account-ada",
                "content": "alpha",
                "source_event_ids": ["event-1"],
            }
        )


def test_compress_memory_proposal_rejects_target_memory_id() -> None:
    with pytest.raises(ValidationError):
        validate_policy_action(
            {
                "action_type": "compress_memory",
                "source_memory_ids": ["mem-1", "mem-2"],
                "target_memory_id": "mem-llm-target",
                "compressed_content": "summary",
            }
        )


def test_compile_generates_stable_memory_ids_for_new_writes() -> None:
    proposals = validate_policy_actions(
        [
            {
                "action_type": "write_memory",
                "entity_id": "account-ada",
                "content": "alpha",
                "source_event_ids": ["event-1"],
            },
            {
                "action_type": "write_memory",
                "entity_id": "account-ada",
                "content": "alpha",
                "source_event_ids": ["event-1"],
            },
        ]
    )

    actions = compile_policy_actions(proposals, active_memory_ids=[])

    assert isinstance(actions[0], WriteMemoryAction)
    assert isinstance(actions[1], WriteMemoryAction)
    # Identical proposals should still yield distinct IDs to avoid SQLite collisions.
    assert actions[0].memory_id != actions[1].memory_id
    assert actions[0].memory_id == deterministic_memory_id(["event-1"], "alpha")


def test_compile_rejects_unknown_memory_id_for_update() -> None:
    proposals = validate_policy_actions(
        [
            {
                "action_type": "update_memory",
                "memory_id": "mem-missing",
                "content": "beta",
                "source_event_ids": ["event-2"],
                "reason": "newer evidence",
            }
        ]
    )

    with pytest.raises(PolicyPlanError) as exc:
        compile_policy_actions(proposals, active_memory_ids=["mem-other"])
    assert "mem-missing" in str(exc.value)
    assert "mem-other" in str(exc.value)


def test_compile_rejects_update_then_mark_stale_in_same_plan() -> None:
    """Reproduces the Ada-style failure: update_memory + mark_stale on same memory."""

    proposals = validate_policy_actions(
        [
            {
                "action_type": "update_memory",
                "memory_id": "mem-ada",
                "content": "Account Ada now prefers email.",
                "source_event_ids": ["event-1"],
                "reason": "correction",
            },
            {
                "action_type": "mark_stale",
                "memory_id": "mem-ada",
                "reason": "redundant",
            },
        ]
    )

    with pytest.raises(PolicyPlanError):
        compile_policy_actions(proposals, active_memory_ids=["mem-ada"])


def test_compile_rejects_mark_stale_then_update_in_same_plan() -> None:
    proposals = validate_policy_actions(
        [
            {
                "action_type": "mark_stale",
                "memory_id": "mem-ada",
                "reason": "old",
            },
            {
                "action_type": "update_memory",
                "memory_id": "mem-ada",
                "content": "later content",
                "source_event_ids": ["event-1"],
                "reason": "should not coexist",
            },
        ]
    )

    with pytest.raises(PolicyPlanError):
        compile_policy_actions(proposals, active_memory_ids=["mem-ada"])


def test_compile_rejects_index_now_for_unknown_memory() -> None:
    proposals = validate_policy_actions(
        [
            {
                "action_type": "index_now",
                "memory_id": "mem-not-yet-existing",
            }
        ]
    )

    with pytest.raises(PolicyPlanError) as exc:
        compile_policy_actions(proposals, active_memory_ids=[])
    assert "index_immediately" in str(exc.value)


def test_compile_rejects_compress_with_unknown_source() -> None:
    proposals = validate_policy_actions(
        [
            {
                "action_type": "compress_memory",
                "source_memory_ids": ["mem-active", "mem-missing"],
                "compressed_content": "summary",
            }
        ]
    )

    with pytest.raises(PolicyPlanError):
        compile_policy_actions(proposals, active_memory_ids=["mem-active"])


def test_compile_compress_assigns_target_memory_id_avoiding_active_collisions() -> None:
    proposals = [
        CompressMemoryProposal(
            source_memory_ids=["mem-a", "mem-b"],
            compressed_content="merged",
        )
    ]

    actions = compile_policy_actions(
        proposals,
        active_memory_ids=["mem-a", "mem-b"],
    )

    assert isinstance(actions[0], CompressMemoryAction)
    assert actions[0].target_memory_id not in {"mem-a", "mem-b"}
    assert actions[0].source_memory_ids == ["mem-a", "mem-b"]


def test_compile_passes_through_existing_memory_actions() -> None:
    proposals = [
        IgnoreEventAction(event_id="event-noise", reason="low-value"),
        IndexNowAction(memory_id="mem-existing"),
        DelayIndexAction(memory_id="mem-existing", retry_after_ms=500, reason="budget"),
        MarkStaleAction(memory_id="mem-other", reason="superseded"),
    ]

    actions = compile_policy_actions(
        proposals,
        active_memory_ids=["mem-existing", "mem-other"],
    )

    assert [type(action) for action in actions] == [
        IgnoreEventAction,
        IndexNowAction,
        DelayIndexAction,
        MarkStaleAction,
    ]


def test_validate_action_plan_does_not_raise_when_plan_is_safe() -> None:
    proposals = [
        _write_proposal(),
        UpdateMemoryAction(
            memory_id="mem-existing",
            content="newer",
            source_event_ids=["event-1"],
            reason="correction",
        ),
    ]
    validate_action_plan(proposals, active_memory_ids=["mem-existing"])
