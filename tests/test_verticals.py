"""Typed category schemas and seeded sampling."""
import random
from pathlib import Path

import pytest

from adaptgym.verticals import VerticalSchema

SHOES = Path(__file__).with_name("fixtures") / "running_shoes.schema.json"


def test_preference_types_units_operators_and_allowed_values():
    vertical = VerticalSchema.load(SHOES)
    price = vertical.assignment("price", 99.5, "lt")
    assert price["unit"] == "EUR"
    assert price["operator"] == "lt"
    vertical.validate_assignment(price)
    assert vertical.assignment("size", 42.5)["unit"] == "EU"
    assert vertical.assignment("brand", ["Nike", "Adidas"], "in")["value"] == ["Nike", "Adidas"]
    for name, value, op in [("brand", "Unknown", "eq"), ("brand", "Nike", "lte"),
                             ("price", "100", "lte"), ("price", True, "eq"),
                             ("price", 99.3, "eq"), ("size", 99, "eq"),
                             ("price", float("nan"), "eq"), ("unknown", "x", "eq"),
                             ("brand", ["Nike", "Nike"], "in")]:
        with pytest.raises(ValueError):
            vertical.assignment(name, value, op)


@pytest.mark.parametrize("changed", [
    {"vertical_id": "other"}, {"schema_version": "2"}, {"unit": "USD"},
    {"extra": "not defined"}, {"attribute": "unknown"},
])
def test_assignment_cannot_change_schema_reference_or_units(changed):
    vertical = VerticalSchema.load(SHOES)
    assignment = vertical.assignment("price", 100, "lte")
    assignment.update(changed)
    with pytest.raises(ValueError):
        vertical.validate_assignment(assignment)


def test_sampling_is_seeded():
    vertical = VerticalSchema.load(SHOES)
    first_rng, second_rng = random.Random(4), random.Random(4)
    for name in vertical.attributes:
        first = vertical.sample(name, first_rng)
        assert first == vertical.sample(name, second_rng)
        vertical.validate_assignment(first)


def test_numeric_sampling_respects_exclusive_bounds_and_decimal_steps():
    schema = VerticalSchema.load(SHOES).schema
    schema["properties"] = {"price": {
        "type": "number", "exclusiveMinimum": 0.1, "exclusiveMaximum": 0.4,
        "multipleOf": 0.1, "x-adaptgym": {"operators": ["lte"]},
    }}
    vertical = VerticalSchema(schema)
    for seed in range(20):
        assignment = vertical.sample("price", random.Random(seed), "lte")
        assert assignment["value"] in (0.2, 0.3)
        assert assignment["operator"] == "lte"


def test_fractional_interval_without_step_is_sampleable():
    schema = VerticalSchema.load(SHOES).schema
    schema["properties"] = {"price": {
        "type": "number", "minimum": 0.1, "maximum": 0.2,
        "x-adaptgym": {"operators": ["eq"]},
    }}
    assert 0.1 <= VerticalSchema(schema).sample("price", random.Random(7))["value"] <= 0.2


def test_unsupported_schema_features_fail_explicitly():
    schema = VerticalSchema.load(SHOES).schema
    schema["properties"]["brand"]["$ref"] = "https://example.com/schema.json"
    with pytest.raises(ValueError, match="unsupported"):
        VerticalSchema(schema)


def test_caller_mutation_does_not_change_loaded_schema():
    schema = VerticalSchema.load(SHOES).schema
    vertical = VerticalSchema(schema)
    schema["properties"]["brand"]["enum"].append("Invented")
    returned = vertical.schema
    returned["properties"]["brand"]["enum"].append("Invented")
    with pytest.raises(ValueError):
        vertical.assignment("brand", "Invented")
