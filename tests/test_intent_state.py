"""Reference state transitions, scoped preferences, and two-way provenance."""
import copy
from pathlib import Path

import pytest

from adaptgym.intent_state import IntentState, empty_state, evidence, validate_evidence, validate_state
from adaptgym.verticals import VerticalSchema


SCHEMA = Path(__file__).with_name("fixtures") / "running_shoes.schema.json"
MESSAGES = [
    {"id": "u0", "role": "user", "text": "I'm starting a running-shoe purchase."},
    {"id": "a1", "role": "assistant", "text": "Is EUR 100 your maximum budget?"},
    {"id": "u1", "role": "user", "text": "Yes."},
    {"id": "u2", "role": "user", "text": "Across purchases, I generally prefer Adidas."},
    {"id": "u3", "role": "user", "text": "Actually, make the maximum EUR 150."},
    {"id": "u4", "role": "user", "text": "Drop that budget requirement."},
    {"id": "u5", "role": "user", "text": "Start a separate purchase; Nike is mandatory for this order."},
    {"id": "u6", "role": "user", "text": "That purchase is finished."},
    {"id": "u7", "role": "user", "text": "On several separate purchases I chose Adidas and was happy."},
]


def prefix(last):
    index = next(i for i, message in enumerate(MESSAGES) if message["id"] == last)
    return MESSAGES[:index + 1]


def record(*ids, mode="explicit"):
    return evidence(mode, list(ids), "The cited user evidence supports the value in its conversational context.")


def event(assignment, refs, *, operation="set", scope="task", strength="hard"):
    return {"operation": operation, "assignment": assignment, "scope": scope,
            "strength": strength, "evidence": [refs]}


def started():
    vertical = VerticalSchema.load(SCHEMA)
    state = IntentState([vertical])
    state.apply({"operation": "begin_task", "task_id": "task-1", "evidence": [record("u0")]}, prefix("u0"))
    return vertical, state


def test_question_plus_confirmation_can_support_a_constraint():
    vertical, state = started()
    state.apply(event(vertical.assignment("price", 100, "lte"), record("a1", "u1")), prefix("u1"))
    entry = state.state["entries"][0]
    assert entry["assignment"]["value"] == 100
    assert entry["evidence"][0]["message_ids"] == ["a1", "u1"]
    assert entry["task_id"] == "task-1"


@pytest.mark.parametrize("refs,last", [
    (record("a1"), "u1"),             # assistant suggestion alone
    (record("u0", "a1"), "u1"),      # no user acceptance after the suggestion
    (record("u3"), "u1"),             # future evidence
    (record("private-event-1"), "u1"),
    (record("u1", "u1"), "u1"),       # duplicate reference
])
def test_invalid_evidence_cannot_mutate_state(refs, last):
    vertical, state = started()
    before = state.state
    with pytest.raises(ValueError):
        state.apply(event(vertical.assignment("price", 100, "lte"), refs), prefix(last))
    assert state.state == before


def test_revise_and_retract_preserve_prior_values_and_transition_evidence():
    vertical, state = started()
    initial = vertical.assignment("price", 100, "lte")
    revised = vertical.assignment("price", 150, "lte")
    state.apply(event(initial, record("a1", "u1")), prefix("u1"))
    state.apply(event(revised, record("a1", "u1", "u3")), prefix("u3"))
    assert state.state["entries"][0]["assignment"] == revised
    history = state.state["history"][0]
    assert history["entry"]["assignment"] == initial
    assert history["status"] == "superseded"
    assert "u3" in history["changed_by"][0]["message_ids"]
    state.apply(event(revised, record("u3", "u4"), operation="remove"), prefix("u4"))
    assert state.state["entries"] == []
    assert state.state["history"][-1]["status"] == "retracted"


def test_task_boundary_expires_temporary_values_but_preserves_durable_preferences():
    vertical, state = started()
    durable = event(vertical.assignment("brand", "Adidas"), record("u2"), scope="durable", strength="soft")
    state.apply(durable, prefix("u2"))
    state.apply(event(vertical.assignment("price", 150, "lte"), record("u3")), prefix("u3"))
    state.apply({"operation": "begin_task", "task_id": "task-2", "evidence": [record("u5")]}, prefix("u5"))
    assert state.state["active_task_id"] == "task-2"
    assert [entry["scope"] for entry in state.state["entries"]] == ["durable"]
    assert state.state["history"][-1]["status"] == "expired"
    state.apply(event(vertical.assignment("brand", "Nike"), record("u5")), prefix("u5"))
    assert len(state.state["entries"]) == 2
    state.apply({"operation": "end_task", "task_id": "task-2", "evidence": [record("u6")]}, prefix("u6"))
    assert state.state["active_task_id"] is None
    assert state.state["entries"][0]["assignment"]["value"] == "Adidas"


def test_mixed_evidence_reinforces_without_overwriting_provenance():
    vertical, state = started()
    assignment = vertical.assignment("brand", "Adidas")
    state.apply(event(assignment, record("u2"), scope="durable", strength="soft"), prefix("u2"))
    state.apply(event(assignment, record("u7", mode="implicit"), operation="reinforce", scope="durable", strength="soft"), prefix("u7"))
    entry = state.state["entries"][0]
    assert entry["evidence_mode"] == "mixed"
    assert len(entry["evidence"]) == 2
    assert state.state["history"] == []


def test_unknown_scope_is_not_promoted_to_durable_or_temporary():
    vertical, state = started()
    state.apply(event(vertical.assignment("size", 42), record("u1"), scope="unknown", strength="unknown"), prefix("u1"))
    entry = state.state["entries"][0]
    assert entry["scope"] == "unknown"
    assert entry["task_id"] is None


def test_returned_snapshots_do_not_mutate_reference_state():
    _, state = started()
    snapshot = state.state
    snapshot["active_task_id"] = "invented"
    assert state.state["active_task_id"] == "task-1"


@pytest.mark.parametrize("operation", ["remove", "reinforce"])
def test_remove_and_reinforce_require_an_existing_value(operation):
    vertical, state = started()
    with pytest.raises(ValueError):
        state.apply(event(vertical.assignment("price", 100, "lte"), record("u1"), operation=operation), prefix("u1"))


def test_task_requirement_needs_an_active_task():
    vertical = VerticalSchema.load(SCHEMA)
    state = IntentState([vertical])
    with pytest.raises(ValueError, match="task ID"):
        state.apply(event(vertical.assignment("price", 100, "lte"), record("u1")), prefix("u1"))
    assert state.state == empty_state()


@pytest.mark.parametrize("refs", [[], [record("missing")], [record("u0", mode="implicit")]])
def test_task_boundaries_require_valid_explicit_evidence_even_with_no_entries(refs):
    state = IntentState([VerticalSchema.load(SCHEMA)])
    with pytest.raises(ValueError):
        state.apply({"operation": "begin_task", "task_id": "task-1", "evidence": refs}, prefix("u0"))
    assert state.state == empty_state()


def test_validation_rejects_schema_drift_and_duplicate_entries():
    vertical, state = started()
    state.apply(event(vertical.assignment("price", 100, "lte"), record("a1", "u1")), prefix("u1"))
    bad = state.state
    bad["entries"][0]["assignment"]["unit"] = "USD"
    with pytest.raises(ValueError):
        validate_state(bad, [vertical], prefix("u1"))
    bad = state.state
    bad["entries"].append(copy.deepcopy(bad["entries"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_state(bad, [vertical], prefix("u1"))


def test_evidence_justification_is_required_but_not_a_semantic_proof():
    bad = record("u0")
    bad["justification"] = ""
    with pytest.raises(ValueError, match="justification"):
        validate_evidence([bad], prefix("u0"))
    # Reference validation checks visibility, not whether the sentence entails a value.
    validate_evidence([record("u0")], prefix("u0"))
