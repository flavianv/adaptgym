"""General persona schema and cross-field coherence validation.

Product preferences belong to vertical schemas, not the persona.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

GENERAL_PERSONA_SCHEMA: dict[str, Any] = json.loads(
    (Path(__file__).with_name("schemas") / "general_persona.schema.json").read_text(encoding="utf-8")
)


def validate_persona(profile: dict[str, Any], *,
                     schema: dict[str, Any] = GENERAL_PERSONA_SCHEMA) -> list[str]:
    """Return schema and cross-field coherence errors for a general persona."""
    errors = [f"schema {'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
              for error in sorted(Draft202012Validator(schema).iter_errors(profile),
                                  key=lambda e: list(map(str, e.absolute_path)))]
    if errors:
        return errors
    first, last = profile.get("first_name", ""), profile.get("last_name", "")
    if profile.get("full_name", "").strip().casefold() != f"{first} {last}".strip().casefold():
        errors.append("full_name must equal first_name followed by last_name")
    income = profile.get("income", {})
    kind = income.get("kind")
    values = {key: income.get(key) for key in ("amount", "minimum", "maximum", "level")}
    if kind == "amount" and not (values["amount"] is not None and values["minimum"] is None
                                  and values["maximum"] is None and values["level"] is None):
        errors.append("income kind 'amount' requires only amount")
    elif kind == "range" and not (values["amount"] is None and values["minimum"] is not None
                                   and values["maximum"] is not None and values["level"] is None
                                   and values["minimum"] <= values["maximum"]):
        errors.append("income kind 'range' requires minimum <= maximum and no amount/level")
    elif kind == "level" and not (values["amount"] is None and values["minimum"] is None
                                   and values["maximum"] is None and isinstance(values["level"], str)):
        errors.append("income kind 'level' requires only a non-empty level")
    family = profile.get("family", {})
    status, partner = family.get("relationship_status"), family.get("partner")
    partnered_statuses = {"dating", "partnered", "married", "separated"}
    if partner is not None and status not in partnered_statuses:
        errors.append(f"relationship status {status!r} conflicts with a current partner")
    if status in {"partnered", "married"} and partner is None:
        errors.append(f"relationship status {status!r} requires a partner record")
    age = profile.get("age", 0)
    if age < 18 and status not in {"single", "dating"}:
        errors.append("for ages 16–17, relationship status must be single or dating")
    children = family.get("children", [])
    child_names: set[str] = set()
    for child in children:
        key = child["first_name"].casefold()
        if key in child_names:
            errors.append(f"duplicate child name {child['first_name']!r}")
        child_names.add(key)
        if age - child["age"] < 16:
            errors.append(f"child {child['first_name']!r} is implausibly close in age to the parent")
    if partner and partner["age"] < 18:
        errors.append("current partner must be an adult")
    household = profile.get("household", {})
    member_keys: set[tuple[str, str]] = set()
    for member in household.get("members", []):
        key = (member["relationship"], member["first_name"].casefold())
        if key in member_keys:
            errors.append(f"duplicate household member {member['first_name']!r}")
        member_keys.add(key)
        if member["relationship"] == "child":
            child = next((c for c in children if c["first_name"].casefold() == member["first_name"].casefold()), None)
            if child is None or child["age"] != member["age"]:
                errors.append(f"household child {member['first_name']!r} does not match family.children")
        if member["relationship"] == "partner":
            if partner is None or partner["first_name"].casefold() != member["first_name"].casefold() or partner["age"] != member["age"]:
                errors.append("household partner does not match family.partner")
    for child in children:
        if child["lives_in_household"]:
            matching = [m for m in household.get("members", [])
                        if m["relationship"] == "child" and m["first_name"].casefold() == child["first_name"].casefold()
                        and m["age"] == child["age"]]
            if not matching:
                errors.append(f"child {child['first_name']!r} lives in household but is missing from household.members")
    return errors
