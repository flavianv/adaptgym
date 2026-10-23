"""Typed product attributes, units, operators, and seeded sampling."""
from __future__ import annotations

import copy
import json
import math
import random
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


OPERATORS = {"eq", "in", "lt", "lte", "gt", "gte"}
ROOT_KEYS = {"$schema", "$id", "title", "description", "type", "properties",
             "additionalProperties", "required", "x-adaptgym"}
FIELD_KEYS = {"title", "description", "type", "enum", "const", "examples", "minimum",
              "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
              "minLength", "maxLength", "pattern", "x-adaptgym"}


class VerticalSchema:
    """Load a scalar-attribute JSON Schema with explicit extraction operators.

    Enum, numeric, boolean, and example-backed string fields are supported.
    References, nested objects, arrays, and cross-field rules are rejected rather
    than silently sampling or validating them incorrectly.
    """

    def __init__(self, schema: dict[str, Any]):
        Draft202012Validator.check_schema(schema)
        self._schema = copy.deepcopy(schema)
        if set(schema) - ROOT_KEYS:
            raise ValueError("unsupported vertical schema keywords")
        if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
            raise ValueError("vertical schema must be an object with additionalProperties=false")
        meta = schema.get("x-adaptgym", {})
        if not isinstance(meta, dict) or not all(isinstance(meta.get(k), str) and meta[k] for k in ("vertical_id", "version")):
            raise ValueError("vertical metadata requires nonempty vertical_id and version")
        self.vertical_id, self.version = meta["vertical_id"], meta["version"]
        fields = schema.get("properties", {})
        if not fields or not set(schema.get("required", [])) <= set(fields):
            raise ValueError("vertical must define attributes and valid required field names")
        for name, field in fields.items():
            if not name or not isinstance(field, dict) or set(field) - FIELD_KEYS:
                raise ValueError(f"{name}: unsupported attribute schema")
            if not isinstance(field.get("type"), str) or field["type"] not in {"string", "integer", "number", "boolean"}:
                raise ValueError(f"{name}: only scalar attributes are supported")
            cfg = field.get("x-adaptgym", {})
            ops = cfg.get("operators") if isinstance(cfg, dict) else None
            if not isinstance(ops, list) or not ops or any(not isinstance(op, str) or op not in OPERATORS for op in ops):
                raise ValueError(f"{name}: define supported operators explicitly")
            if len(ops) != len(set(ops)):
                raise ValueError(f"{name}: duplicate operators")
            if field["type"] not in {"number", "integer"} and set(ops) - {"eq", "in"}:
                raise ValueError(f"{name}: ordered comparisons require a numeric attribute")
            if "unit" in cfg and (not isinstance(cfg["unit"], str) or not cfg["unit"]):
                raise ValueError(f"{name}: unit must be a nonempty string")
            for value in field.get("examples", []):
                self._check_value(name, value)

    @classmethod
    def load(cls, path: str | Path) -> "VerticalSchema":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def schema(self) -> dict[str, Any]:
        return copy.deepcopy(self._schema)

    @property
    def attributes(self) -> tuple[str, ...]:
        return tuple(self._schema["properties"])

    def _field(self, name: str) -> dict[str, Any]:
        if not isinstance(name, str) or name not in self._schema["properties"]:
            raise ValueError(f"unknown attribute {name!r} in {self.vertical_id}")
        return self._schema["properties"][name]

    def _check_value(self, name: str, value: Any) -> None:
        field = self._field(name)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name}: nonfinite numbers are invalid")
        # Decimal arithmetic avoids binary-float errors for prices/multipleOf.
        clean = {k: v for k, v in field.items() if k != "multipleOf"}
        errors = list(Draft202012Validator(clean).iter_errors(value))
        if errors:
            raise ValueError(f"{name}: {errors[0].message}")
        if "multipleOf" in field and Decimal(str(value)) % Decimal(str(field["multipleOf"])) != 0:
            raise ValueError(f"{name}: value is not a multiple of {field['multipleOf']}")

    def assignment(self, attribute: str, value: Any, operator: str = "eq") -> dict[str, Any]:
        field = self._field(attribute)
        cfg = field["x-adaptgym"]
        if operator not in cfg["operators"]:
            raise ValueError(f"{attribute}: unsupported operator {operator!r}")
        if operator == "in":
            if not isinstance(value, list) or not value:
                raise ValueError(f"{attribute}: 'in' requires a nonempty value list")
            if len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
                raise ValueError(f"{attribute}: duplicate allowed values")
            for candidate in value:
                self._check_value(attribute, candidate)
        else:
            self._check_value(attribute, value)
        result = {"vertical_id": self.vertical_id, "schema_version": self.version,
                  "attribute": attribute, "operator": operator, "value": copy.deepcopy(value)}
        if "unit" in cfg:
            result["unit"] = cfg["unit"]
        return result

    def validate_assignment(self, assignment: dict[str, Any]) -> None:
        required = {"vertical_id", "schema_version", "attribute", "operator", "value"}
        if not isinstance(assignment, dict) or not required <= set(assignment) or set(assignment) - required - {"unit"}:
            raise ValueError("invalid preference assignment fields")
        if assignment["vertical_id"] != self.vertical_id or assignment["schema_version"] != self.version:
            raise ValueError("assignment references the wrong vertical or schema version")
        expected = self.assignment(assignment["attribute"], assignment["value"], assignment["operator"])
        if assignment != expected:
            raise ValueError("assignment unit must match the vertical schema")

    def sample(self, attribute: str, rng: random.Random, operator: str = "eq") -> dict[str, Any]:
        """Sample a validated value using finite choices or bounded numeric fields."""
        field = self._field(attribute)
        choices = ([field["const"]] if "const" in field else
                   field.get("enum") or field.get("examples") or
                   ([False, True] if field["type"] == "boolean" else None))
        for _ in range(100):
            if choices:
                value = copy.deepcopy(rng.choice(choices))
            elif field["type"] in {"number", "integer"}:
                lo = field.get("minimum", field.get("exclusiveMinimum"))
                hi = field.get("maximum", field.get("exclusiveMaximum"))
                if lo is None or hi is None:
                    raise ValueError(f"{attribute}: sampling requires examples or finite numeric bounds")
                if field["type"] == "number" and "multipleOf" not in field:
                    value = rng.uniform(lo, hi)
                    try:
                        return self.assignment(attribute, [value] if operator == "in" else value, operator)
                    except ValueError:
                        if operator not in field["x-adaptgym"]["operators"]:
                            raise
                        continue
                lower, upper = Decimal(str(lo)), Decimal(str(hi))
                step = Decimal(str(field.get("multipleOf", 1)))
                if field["type"] == "integer":
                    # Integer sampling supports integer steps; use examples for fractional steps.
                    if step != step.to_integral_value():
                        raise ValueError(f"{attribute}: integer sampling with fractional multipleOf requires examples")
                first = int((lower / step).to_integral_value(rounding=ROUND_CEILING))
                last = int((upper / step).to_integral_value(rounding=ROUND_FLOOR))
                if "exclusiveMinimum" in field:
                    first = max(first, int((Decimal(str(field["exclusiveMinimum"])) / step).to_integral_value(rounding=ROUND_FLOOR)) + 1)
                if "exclusiveMaximum" in field:
                    last = min(last, int((Decimal(str(field["exclusiveMaximum"])) / step).to_integral_value(rounding=ROUND_CEILING)) - 1)
                if first > last:
                    raise ValueError(f"{attribute}: no sampleable values within bounds")
                sampled = Decimal(rng.randint(first, last)) * step
                value = int(sampled) if field["type"] == "integer" else float(sampled)
            else:
                raise ValueError(f"{attribute}: string sampling requires enum, const, or examples")
            try:
                return self.assignment(attribute, [value] if operator == "in" else value, operator)
            except ValueError:
                if operator not in field["x-adaptgym"]["operators"]:
                    raise
        raise ValueError(f"{attribute}: could not sample a valid value")
