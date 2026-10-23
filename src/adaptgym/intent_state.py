"""Typed, scoped intent state with observable evidence and revision history."""
from __future__ import annotations

import copy
from typing import Any

from adaptgym.verticals import VerticalSchema


def empty_state() -> dict[str, Any]:
    return {"active_task_id": None, "entries": [], "history": []}


def evidence(mode: str, message_ids: list[str], justification: str) -> dict[str, Any]:
    return {"source_type": "conversation", "mode": mode,
            "message_ids": list(message_ids), "justification": justification}


def entry_key(entry: dict[str, Any]) -> tuple:
    assignment = entry["assignment"]
    return (assignment["vertical_id"], assignment["schema_version"], assignment["attribute"],
            entry["scope"], entry["task_id"])


def evidence_mode(records: list[dict[str, Any]]) -> str:
    if not records:
        raise ValueError("supporting evidence must not be empty")
    modes = {record["mode"] for record in records}
    return "mixed" if len(modes) > 1 else next(iter(modes))


def validate_evidence_messages(messages: list[dict[str, Any]]) -> None:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    ids = []
    for message in messages:
        if (not isinstance(message, dict) or not isinstance(message.get("id"), str) or not message["id"]
                or message.get("role") not in ("user", "assistant") or not isinstance(message.get("text"), str)):
            raise ValueError("messages require an ID, user/assistant role, and text")
        ids.append(message["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate message IDs")


def validate_evidence(records: list[dict[str, Any]], messages: list[dict[str, Any]]) -> None:
    """Allow assistant context, but require user evidence after that context."""
    validate_evidence_messages(messages)
    observed = {message["id"]: (index, message) for index, message in enumerate(messages)}
    if not isinstance(records, list) or not records:
        raise ValueError("every value or transition needs supporting evidence")
    for record in records:
        if not isinstance(record, dict) or set(record) != {"source_type", "mode", "message_ids", "justification"}:
            raise ValueError("invalid evidence record")
        if record["source_type"] != "conversation" or record["mode"] not in ("explicit", "implicit"):
            raise ValueError("invalid evidence source or mode")
        ids = record["message_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(mid, str) for mid in ids) or len(set(ids)) != len(ids):
            raise ValueError("evidence must reference distinct message IDs")
        if any(mid not in observed for mid in ids):
            raise ValueError("evidence must not reference future or private records")
        cited = [observed[mid] for mid in ids]
        if max(cited, key=lambda item: item[0])[1]["role"] != "user":
            raise ValueError("assistant context must be followed by cited user evidence")
        if not isinstance(record["justification"], str) or not record["justification"].strip() or len(record["justification"]) > 500:
            raise ValueError("evidence needs a concise justification (1–500 characters)")


def validate_state(state: dict[str, Any], verticals: list[VerticalSchema],
                   messages: list[dict[str, Any]]) -> None:
    """Reject invented attributes, bad scope, and hidden/future evidence references.

    This checks the output contract and reference visibility. Semantic support is
    scored against episode evidence or by an independent reader, not inferred
    merely from the existence of a cited message.
    """
    if not isinstance(state, dict) or set(state) != {"active_task_id", "entries", "history"}:
        raise ValueError("state requires active_task_id, entries, and history")
    active = state["active_task_id"]
    if active is not None and (not isinstance(active, str) or not active):
        raise ValueError("active_task_id must be a nonempty string or null")
    if not isinstance(state["entries"], list) or not isinstance(state["history"], list):
        raise ValueError("entries and history must be lists")
    registry = {vertical.vertical_id: vertical for vertical in verticals}
    if len(registry) != len(verticals):
        raise ValueError("duplicate vertical IDs")
    validate_evidence_messages(messages)

    def check_evidence(records):
        validate_evidence(records, messages)

    def check_entry(entry):
        required = {"assignment", "scope", "task_id", "strength", "evidence_mode", "evidence"}
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError("invalid intent entry fields")
        assignment = entry["assignment"]
        if not isinstance(assignment, dict) or not isinstance(assignment.get("vertical_id"), str) or assignment["vertical_id"] not in registry:
            raise ValueError("unknown assignment vertical")
        registry[assignment["vertical_id"]].validate_assignment(assignment)
        if entry["scope"] not in ("durable", "task", "unknown") or entry["strength"] not in ("soft", "hard", "unknown"):
            raise ValueError("invalid scope or strength")
        if entry["scope"] == "task":
            if not isinstance(entry["task_id"], str) or not entry["task_id"]:
                raise ValueError("task entries require a task ID")
        elif entry["task_id"] is not None:
            raise ValueError("durable and unknown-scope entries must not claim a task ID")
        check_evidence(entry["evidence"])
        if entry["evidence_mode"] != evidence_mode(entry["evidence"]):
            raise ValueError("evidence_mode must summarize all supporting evidence")

    keys = []
    for entry in state["entries"]:
        check_entry(entry)
        if entry["scope"] == "task" and entry["task_id"] != active:
            raise ValueError("active task entries must belong to the current task")
        keys.append(entry_key(entry))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate intent entries")
    for change in state["history"]:
        if not isinstance(change, dict) or set(change) != {"entry", "status", "changed_by"}:
            raise ValueError("invalid history record")
        if change["status"] not in ("superseded", "retracted", "expired"):
            raise ValueError("invalid history status")
        check_entry(change["entry"])
        check_evidence(change["changed_by"])


class IntentState:
    """Reference transition engine for simulator ground truth, not an extractor."""

    def __init__(self, verticals: list[VerticalSchema]):
        self.verticals = verticals
        self._state = empty_state()
        validate_state(self._state, verticals, [])

    @property
    def state(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def apply(self, event: dict[str, Any], messages: list[dict[str, Any]]) -> None:
        """Apply a private generation event atomically against visible evidence."""
        state = self.state
        operation, records = event["operation"], copy.deepcopy(event["evidence"])
        validate_evidence(records, messages)
        if operation in {"begin_task", "end_task"}:
            task = event.get("task_id")
            if operation == "begin_task" and (not isinstance(task, str) or not task or task == state["active_task_id"]):
                raise ValueError("begin_task requires a new task ID")
            if operation == "end_task" and (state["active_task_id"] is None or task != state["active_task_id"]):
                raise ValueError("end_task must match the active task")
            kept = []
            for entry in state["entries"]:
                if entry["scope"] == "task":
                    state["history"].append({"entry": entry, "status": "expired", "changed_by": records})
                else:
                    kept.append(entry)
            state["entries"] = kept
            state["active_task_id"] = task if operation == "begin_task" else None
        elif operation in {"set", "reinforce", "remove"}:
            assignment = copy.deepcopy(event["assignment"])
            scope = event["scope"]
            task = state["active_task_id"] if scope == "task" else None
            entry = {"assignment": assignment, "scope": scope, "task_id": task,
                     "strength": event["strength"], "evidence_mode": evidence_mode(records), "evidence": records}
            key = entry_key(entry)
            old = next((item for item in state["entries"] if entry_key(item) == key), None)
            if operation in {"remove", "reinforce"} and old is None:
                raise ValueError(f"{operation} requires an existing entry")
            if operation == "reinforce":
                if assignment != old["assignment"] or event["strength"] != old["strength"]:
                    raise ValueError("reinforcement must preserve the value and strength")
                old["evidence"].extend(record for record in records if record not in old["evidence"])
                old["evidence_mode"] = evidence_mode(old["evidence"])
            else:
                if operation == "remove" and (assignment != old["assignment"] or event["strength"] != old["strength"]):
                    raise ValueError("removal must identify the active value and strength")
                state["entries"] = [item for item in state["entries"] if entry_key(item) != key]
                if old:
                    state["history"].append({"entry": old, "status": "retracted" if operation == "remove" else "superseded",
                                             "changed_by": records})
                if operation == "set":
                    state["entries"].append(entry)
        else:
            raise ValueError(f"unknown intent operation {operation!r}")
        if operation in {"begin_task", "end_task"} and any(record["mode"] != "explicit" for record in records):
            raise ValueError("task boundaries require observable explicit evidence")
        validate_state(state, self.verticals, messages)
        self._state = state
