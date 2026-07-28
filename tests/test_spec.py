"""Tests for the ChartSpec seam.

The spec is the contract an agent writes against, so these tests care less
about internal plumbing than about the two properties a caller depends on:
a malformed spec fails loudly with an actionable message, and a valid spec
round-trips through JSON unchanged.
"""

from __future__ import annotations

import json

import pytest

from dataviz_agent.errors import SpecError
from dataviz_agent.spec import (
    CHART_TYPES,
    ChartSpec,
    Encoding,
    Filter,
    chart_catalog,
)


# -- encodings ---------------------------------------------------------------


def test_bare_column_name_is_a_legal_encoding():
    spec = ChartSpec(chart="column", x="region", y="revenue")
    assert spec.x == Encoding(field="region")


def test_encoding_rejects_unknown_key_with_a_suggestion():
    with pytest.raises(SpecError, match="did you mean: aggregate"):
        Encoding.coerce({"field": "revenue", "aggregat": "sum"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"field": ""},
        {"field": "x", "aggregate": "avg"},  # 'mean', not 'avg'
        {"field": "x", "time_unit": "fortnight"},
        {"field": "x", "sort": "ascending"},
        {"field": "x", "scale": "sqrt"},
        {"field": "x", "bins": 0},
    ],
)
def test_encoding_validates_its_fields(kwargs):
    with pytest.raises(SpecError):
        Encoding(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"field": "net_rev_usd"}, "Net rev usd"),
        ({"field": "revenue", "aggregate": "sum"}, "Sum of Revenue"),
        ({"field": "order_id", "aggregate": "count"}, "Count"),
        ({"field": "ordered_at", "time_unit": "month"}, "Ordered at (month)"),
        ({"field": "revenue", "aggregate": "sum", "title": "Net"}, "Net"),
    ],
)
def test_encoding_label(kwargs, expected):
    assert Encoding(**kwargs).label == expected


# -- filters -----------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"field": "region", "op": "~="},
        {"field": "region", "op": "in", "value": "west"},  # needs a sequence
        {"field": "n", "op": "between", "value": [1, 2, 3]},
        {"field": "n", "op": "isnull", "value": 3},
    ],
)
def test_filter_validates(kwargs):
    with pytest.raises(SpecError):
        Filter(**kwargs)


def test_filter_accepts_a_mapping():
    assert Filter.coerce({"field": "region", "op": "==", "value": "West"}).value == "West"


# -- chart-level validation --------------------------------------------------


def test_unknown_chart_type_suggests_a_real_one():
    with pytest.raises(SpecError, match="did you mean: scatter"):
        ChartSpec(chart="scattr", x="a", y="b")


def test_missing_required_channel_names_the_channel():
    with pytest.raises(SpecError, match="requires the color channel"):
        ChartSpec(chart="grouped_bar", x="region", y="revenue")


def test_unused_channel_is_rejected_rather_than_dropped():
    with pytest.raises(SpecError, match="does not use the size channel"):
        ChartSpec(chart="heatmap", x="a", y="b", color="c", size="d")


def test_normalize_without_stack_is_an_error():
    with pytest.raises(SpecError, match="normalize"):
        ChartSpec(chart="stacked_bar", x="a", y="b", color="c", normalize=True)


@pytest.mark.parametrize(
    "kwargs",
    [{"limit": 0}, {"mode": "sepia"}, {"orientation": "sideways"}, {"width": 40}],
)
def test_spec_validates_scalars(kwargs):
    with pytest.raises(SpecError):
        ChartSpec(chart="column", x="a", y="b", **kwargs)


def test_every_registry_entry_can_be_instantiated():
    """The registry drives validation, so every entry must be satisfiable."""
    for name, meta in CHART_TYPES.items():
        channels = {c: f"col_{c}" for c in meta.required}
        spec = ChartSpec(chart=name, **channels)
        assert spec.meta is meta


# -- derived properties ------------------------------------------------------


def test_emphasis_overrides_the_forms_color_job():
    spec = ChartSpec(chart="column", x="region", y="revenue", emphasis="West")
    assert spec.meta.color_job == "sequential"
    assert spec.color_job == "emphasis"


def test_orientation_auto_resolves_per_form():
    assert ChartSpec(chart="bar", x="a", y="b").is_horizontal
    assert not ChartSpec(chart="column", x="a", y="b").is_horizontal
    assert not ChartSpec(chart="bar", x="a", y="b", orientation="vertical").is_horizontal


def test_fields_used_covers_channels_and_filters_without_duplicates():
    spec = ChartSpec(
        chart="line",
        x="date",
        y={"field": "revenue", "aggregate": "sum"},
        color="region",
        filters=[{"field": "region", "op": "!=", "value": "Other"}],
    )
    assert spec.fields_used() == ["date", "revenue", "region"]


def test_with_note_deduplicates_and_leaves_the_original_alone():
    spec = ChartSpec(chart="column", x="a", y="b")
    noted = spec.with_note("top 10 applied").with_note("top 10 applied")
    assert noted.notes == ("top 10 applied",)
    assert spec.notes == ()


def test_with_changes_revalidates():
    spec = ChartSpec(chart="column", x="a", y="b")
    with pytest.raises(SpecError):
        spec.with_changes(chart="grouped_bar")  # now missing `color`


# -- serialization -----------------------------------------------------------


def test_to_dict_is_minimal_and_leads_with_chart():
    spec = ChartSpec(chart="column", x="region", y="revenue")
    data = spec.to_dict()
    assert list(data) == ["chart", "x", "y"]
    assert data["x"] == "region"  # bare name, not a dict of nulls


def test_to_dict_keeps_non_default_encoding_keys():
    spec = ChartSpec(chart="column", x="region", y={"field": "revenue", "aggregate": "sum"})
    assert spec.to_dict()["y"] == {"field": "revenue", "aggregate": "sum"}


def test_round_trip_through_json():
    spec = ChartSpec(
        chart="line",
        x={"field": "ordered_at", "time_unit": "month"},
        y={"field": "revenue", "aggregate": "sum"},
        color="region",
        filters=[{"field": "region", "op": "in", "value": ["West", "East"]}],
        limit=10,
        mode="dark",
        title="Revenue by region",
        notes=["top 10 applied"],
    )
    assert ChartSpec.from_json(spec.to_json()) == spec


def test_round_trip_is_stable_across_repeated_serialization():
    spec = ChartSpec(chart="stacked_bar", x="a", y="b", color="c", stack=True, normalize=True)
    once = spec.to_json()
    assert ChartSpec.from_json(once).to_json() == once


def test_from_dict_rejects_unknown_keys_with_a_suggestion():
    with pytest.raises(SpecError, match="did you mean: title"):
        ChartSpec.from_dict({"chart": "column", "x": "a", "y": "b", "titel": "T"})


def test_from_json_reports_bad_json_as_a_spec_error():
    with pytest.raises(SpecError, match="not valid JSON"):
        ChartSpec.from_json("{not json")


def test_describe_mentions_every_bound_channel():
    line = ChartSpec(chart="line", x="date", y="revenue", color="region").describe()
    assert "x=Date" in line and "y=Revenue" in line and "color=Region" in line


# -- catalog -----------------------------------------------------------------


def test_catalog_is_json_serializable_and_covers_the_registry():
    catalog = chart_catalog()
    assert {entry["chart"] for entry in catalog} == set(CHART_TYPES)
    json.dumps(catalog)  # must not raise


def test_discouraged_forms_explain_the_alternative():
    for entry in chart_catalog():
        if entry["discouraged"]:
            assert len(entry["discouraged"]) > 20
