"""Configuration integrity tests."""

from __future__ import annotations

from collections.abc import Hashable

import yaml

from quant_equity.config import DEFAULT_CONFIG_PATH, load_config


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Hashable, object]:
    mapping: dict[Hashable, object] = {}

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)

        if key in mapping:
            raise AssertionError(f"Duplicate YAML mapping key detected: {key!r}")

        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def test_project_configuration_contains_no_duplicate_mapping_keys() -> None:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as file:
        loaded = yaml.load(file, Loader=_UniqueKeyLoader)

    assert isinstance(loaded, dict)


def test_portfolio_construction_configuration_is_complete_and_consistent() -> None:
    load_config.cache_clear()
    portfolio_config = load_config()["portfolio_construction"]

    expected_keys = {
        "primary_model_name",
        "challenger_model_name",
        "momentum_model_name",
        "top_n",
        "score_weighted_candidate_count",
        "candidate_count",
        "equal_weight_positions",
        "max_weight",
        "max_security_weight",
        "max_sector_weight",
        "minimum_positions",
        "minimum_cross_section_size",
        "weight_tolerance",
        "optimization_tolerance",
        "optimization_max_iterations",
    }

    assert expected_keys.issubset(portfolio_config)
    assert portfolio_config["score_weighted_candidate_count"] == portfolio_config["candidate_count"]
    assert portfolio_config["max_weight"] == portfolio_config["max_security_weight"]
    assert portfolio_config["max_sector_weight"] == 0.25
