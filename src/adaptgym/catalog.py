"""Fixed catalog text with procedurally controlled, schema-valid attributes."""
from __future__ import annotations

import copy
from decimal import Decimal
from string import Formatter
from typing import Any

from adaptgym.verticals import VerticalSchema


def validate_catalog(catalog: dict[str, Any], vertical: VerticalSchema) -> None:
    if catalog.get("vertical_id") != vertical.vertical_id or catalog.get("schema_version") != vertical.version:
        raise ValueError("catalog schema reference does not match vertical")
    products = catalog.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError("catalog needs nonempty products")
    ids = set()
    for product in products:
        if not isinstance(product, dict) or not isinstance(product.get("id"), str) or not product["id"] or product["id"] in ids:
            raise ValueError("catalog product IDs must be nonempty and unique")
        ids.add(product["id"])
        if not isinstance(product.get("name"), str) or not product["name"]:
            raise ValueError("product needs a name")
        attributes = product.get("attributes")
        if not isinstance(attributes, dict) or set(attributes) != set(vertical.attributes):
            raise ValueError("debug products must define exactly the vertical attributes")
        for attribute, value in attributes.items():
            vertical._check_value(attribute, value)
        if "description" in product and not isinstance(product["description"], str):
            raise ValueError("product description must be text")


def _format(template: str, values: dict[str, Any]) -> str:
    if not isinstance(template, str):
        raise ValueError("catalog text templates must be strings")
    for _, field, _, conversion in Formatter().parse(template):
        if field is not None and (field not in values or conversion):
            raise ValueError(f"unsupported template field {field!r}; use direct attribute names")
    try:
        return template.format_map(values)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"invalid catalog template: {exc}") from exc


def build_catalog(template: dict[str, Any], vertical: VerticalSchema,
                  group_overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Expand groups into SKUs. Overrides alter values, never the text templates.

    An attribute specification is a literal scalar, {start, step}, or
    {values: [...]}; sequence values cycle within a group. Numbers use decimal
    arithmetic so template recipes do not introduce float step artifacts.
    """
    if template.get("vertical_id") != vertical.vertical_id or template.get("schema_version") != vertical.version:
        raise ValueError("template schema reference does not match vertical")
    groups = template.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("template needs nonempty groups")
    names = [group.get("name") for group in groups]
    if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        raise ValueError("group names must be unique nonempty strings")
    overrides = group_overrides or {}
    if set(overrides) - set(names):
        raise ValueError("override references an unknown group")
    variable_attributes = template.get("variable_attributes")
    if variable_attributes is not None:
        if not isinstance(variable_attributes, list) or set(variable_attributes) - set(vertical.attributes):
            raise ValueError("variable_attributes must name schema attributes")
        for values in overrides.values():
            if set(values) - set(variable_attributes):
                raise ValueError("overrides may change only variable_attributes; model properties are fixed")
    products = []
    for group in groups:
        count = group.get("count")
        if type(count) is not int or not 1 <= count <= 10000:
            raise ValueError("group count must be 1–10000")
        specs = {**group.get("attributes", {}), **overrides.get(group["name"], {})}
        if set(specs) != set(vertical.attributes):
            raise ValueError("group must specify exactly the vertical attributes")
        for index in range(count):
            attributes = {}
            for name, spec in specs.items():
                if isinstance(spec, dict):
                    if set(spec) == {"values"} and isinstance(spec["values"], list) and spec["values"]:
                        value = spec["values"][index % len(spec["values"])]
                    elif set(spec) == {"start", "step"}:
                        if any(type(spec[key]) not in (int, float) for key in spec):
                            raise ValueError("start and step must be numbers")
                        number = Decimal(str(spec["start"])) + Decimal(str(spec["step"])) * index
                        value = int(number) if number == number.to_integral_value() else float(number)
                    else:
                        raise ValueError(f"unsupported value specification for {name}")
                else:
                    value = spec
                attributes[name] = copy.deepcopy(value)
            context = {**attributes, "product_number": len(products) + 1, "group": group["name"]}
            products.append({
                "id": _format(template["id_template"], context),
                "name": _format(template["name_template"], context),
                "description": _format(group.get("description_template", template["description_template"]), context),
                "attributes": attributes,
            })
            if "provenance" in group:
                products[-1]["provenance"] = copy.deepcopy(group["provenance"])
    catalog = {"catalog_id": template["catalog_id"], "vertical_id": vertical.vertical_id,
               "schema_version": vertical.version, "synthetic": True, "products": products}
    validate_catalog(catalog, vertical)
    return catalog


def matches(product: dict[str, Any], requirements: list[dict[str, Any]]) -> bool:
    for requirement in requirements:
        name, op, value = requirement["attribute"], requirement["operator"], requirement["value"]
        if name not in product["attributes"]:
            return False
        actual = product["attributes"][name]
        if op == "eq":
            good = actual == value
        elif op == "in":
            good = actual in value
        elif op == "lt":
            good = actual < value
        elif op == "lte":
            good = actual <= value
        elif op == "gt":
            good = actual > value
        elif op == "gte":
            good = actual >= value
        else:
            raise ValueError(f"unknown comparison {op!r}")
        if not good:
            return False
    return True


def matching_ids(catalog: dict[str, Any], requirements: list[dict[str, Any]]) -> list[str]:
    return [product["id"] for product in catalog["products"] if matches(product, requirements)]


def classify_products(catalog: dict[str, Any], hard: list[dict[str, Any]],
                      soft: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Request-relative roles; pool membership never makes an item unsuitable."""
    result = {}
    for product in catalog['products']:
        failed_hard = [r for r in hard if not matches(product, [r])]
        missed_soft = [r for r in soft if not matches(product, [r])]
        role = 'hard_mismatch' if failed_hard else 'soft_miss' if missed_soft else 'suitable'
        result[product['id']] = {'role': role, 'failed_hard': copy.deepcopy(failed_hard),
                                 'missed_soft': copy.deepcopy(missed_soft)}
    return result
