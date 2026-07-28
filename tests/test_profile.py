"""Tests for profiling.

These are mostly tests of judgement, not of plumbing: a store id must not be
classified as something to average, a customer name must not become a bar
chart axis, and a date stored as text must still be a date.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dataviz_agent.errors import IngestError, SpecError
from dataviz_agent.profile import (
    MAX_CATEGORY_CARDINALITY,
    ColumnProfile,
    profile_frame,
)


@pytest.fixture
def sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ordered_at": pd.date_range("2024-01-01", periods=12, freq="ME"),
            "region": ["West", "East", "North"] * 4,
            "revenue": [100.5, 220.0, -15.25, 310.0] * 3,
            "units": [1, 2, 3, 4] * 3,
            "store_id": [11, 12, 13, 14] * 3,
            "is_promo": [True, False] * 6,
            "order_ref": [f"ORD-{i:04d}" for i in range(12)],
        }
    )


def role(df: pd.DataFrame, name: str) -> str:
    return profile_frame(df).column(name).role


# -- role classification -----------------------------------------------------


def test_roles_of_a_typical_sales_table(sales):
    profile = profile_frame(sales)
    assert profile.column("ordered_at").role == "temporal"
    assert profile.column("region").role == "categorical"
    assert profile.column("revenue").role == "quantitative"
    assert profile.column("is_promo").role == "boolean"
    assert profile.column("order_ref").role == "identifier"


def test_numeric_code_column_is_not_a_measure(sales):
    """Averaging a store id is a bug a dtype check never catches."""
    store = profile_frame(sales).column("store_id")
    assert store.role == "categorical"
    assert not store.is_measure
    assert store.notes  # says why


def test_plain_numeric_count_stays_a_measure(sales):
    assert profile_frame(sales).column("units").role == "quantitative"


def test_zero_one_column_is_a_flag():
    df = pd.DataFrame({"churned": [0, 1, 1, 0, 1]})
    assert role(df, "churned") == "boolean"


def test_dates_stored_as_text_are_parsed():
    df = pd.DataFrame({"day": ["2024-01-01", "2024-01-02", "2024-01-03"] * 4})
    col = profile_frame(df).column("day")
    assert col.role == "temporal"
    assert col.min.startswith("2024-01-01")


def test_mostly_text_column_with_a_few_dates_is_not_temporal():
    df = pd.DataFrame({"note": ["2024-01-01", "shipped late", "no idea", "returned"] * 3})
    assert role(df, "note") != "temporal"


def test_numbers_stored_as_text_are_recovered():
    df = pd.DataFrame({"amount": [f"${v:,.2f}" for v in range(1000, 1030)]})
    col = profile_frame(df).column("amount")
    assert col.role == "quantitative"
    assert col.min == 1000.0


def test_free_text_is_not_a_category():
    df = pd.DataFrame(
        {"comment": [f"The customer wrote in about issue number {i} at length" * 2 for i in range(30)]}
    )
    assert role(df, "comment") == "text"


def test_unique_strings_are_identifiers_not_categories():
    df = pd.DataFrame({"customer": [f"Customer {i}" for i in range(40)]})
    col = profile_frame(df).column("customer")
    assert col.role == "identifier"
    assert not col.is_dimension


def test_all_null_column_is_flagged_rather_than_guessed():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
    col = profile_frame(df).column("b")
    assert col.role == "empty"
    assert col.n_missing == 3


# -- statistics --------------------------------------------------------------


def test_measure_statistics(sales):
    revenue = profile_frame(sales).column("revenue")
    assert revenue.min == -15.25
    assert revenue.max == 310.0
    assert revenue.has_negative  # drives the diverging/zero-baseline decision
    assert revenue.mean == pytest.approx(sales["revenue"].mean())


def test_missing_values_are_counted_not_dropped_silently():
    df = pd.DataFrame({"revenue": [1.0, None, 3.0, None]})
    col = profile_frame(df).column("revenue")
    assert col.n_missing == 2
    assert col.missing_ratio == 0.5
    assert col.n_unique == 2


def test_monotonic_time_is_detected(sales):
    assert profile_frame(sales).column("ordered_at").is_monotonic
    shuffled = sales.iloc[::-1]
    assert not profile_frame(shuffled).column("ordered_at").is_monotonic


def test_top_values_are_ranked_for_categories(sales):
    top = profile_frame(sales).column("region").top_values
    assert top[0][1] == 4
    assert {v for v, _ in top} == {"West", "East", "North"}


def test_high_cardinality_category_carries_a_warning():
    df = pd.DataFrame({"sku": [f"sku-{i % 80}" for i in range(200)]})
    col = profile_frame(df).column("sku")
    assert col.role == "categorical"
    assert col.n_unique > MAX_CATEGORY_CARDINALITY
    assert not col.is_drawable_axis  # needs a top-N or a table first
    assert any("top-N" in n for n in col.notes)


# -- frame-level -------------------------------------------------------------


def test_n_rows_is_the_real_count_even_when_sampled():
    df = pd.DataFrame({"n": range(2000), "g": ["a", "b"] * 1000})
    profile = profile_frame(df, sample=500)
    assert profile.n_rows == 2000
    assert profile.truncated


def test_small_frames_are_not_marked_truncated(sales):
    assert not profile_frame(sales).truncated


def test_measures_and_dimensions_split(sales):
    profile = profile_frame(sales)
    assert [c.name for c in profile.measures] == ["revenue", "units"]
    dims = [c.name for c in profile.dimensions]
    assert dims[0] == "ordered_at"  # time first
    assert set(dims) == {"ordered_at", "region", "store_id", "is_promo"}


def test_dimensions_are_ordered_by_cardinality():
    df = pd.DataFrame(
        {"many": [f"m{i % 20}" for i in range(60)], "few": ["a", "b"] * 30}
    )
    assert [c.name for c in profile_frame(df).dimensions] == ["few", "many"]


def test_duplicate_columns_are_rejected():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(IngestError, match="duplicate column"):
        profile_frame(df)


def test_non_dataframe_input_is_an_ingest_error():
    with pytest.raises(IngestError, match="DataFrame"):
        profile_frame({"a": [1, 2]})


# -- lookup ------------------------------------------------------------------


def test_unknown_column_error_lists_the_schema_and_suggests(sales):
    with pytest.raises(SpecError, match="did you mean: revenue"):
        profile_frame(sales).column("revenu")


def test_resolve_tolerates_case_and_separators(sales):
    profile = profile_frame(sales)
    for spelling in ("ordered_at", "Ordered At", "orderedat", "ORDERED_AT"):
        assert profile.resolve(spelling).name == "ordered_at"


def test_resolve_prefers_an_exact_match():
    df = pd.DataFrame({"Region": ["a"], "region": ["b"]})
    profile = profile_frame(df)
    assert profile.resolve("region").name == "region"
    with pytest.raises(SpecError, match="more than one column"):
        profile.resolve("REGION")


def test_membership_and_iteration(sales):
    profile = profile_frame(sales)
    assert "region" in profile and "nope" not in profile
    assert len(profile) == len(sales.columns)
    assert all(isinstance(c, ColumnProfile) for c in profile)


# -- reporting ---------------------------------------------------------------


def test_to_dict_is_json_safe(sales):
    import json

    json.dumps(profile_frame(sales).to_dict())


def test_describe_mentions_every_column(sales):
    text = profile_frame(sales).describe()
    assert all(name in text for name in sales.columns)
    assert "12 rows" in text
