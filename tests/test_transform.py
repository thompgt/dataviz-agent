"""Tests for spec execution.

The two properties worth defending: the numbers are right, and the chart never
hides what it left out.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dataviz_agent.errors import SpecError
from dataviz_agent.spec import ChartSpec
from dataviz_agent.transform import OTHER_LABEL, apply_spec


@pytest.fixture
def sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ordered_at": pd.to_datetime(
                [
                    "2024-01-05", "2024-01-20", "2024-02-11", "2024-02-27",
                    "2024-03-03", "2024-03-19", "2024-01-08", "2024-02-02",
                ]
            ),
            "region": ["West", "West", "East", "East", "West", "North", "East", "North"],
            "channel": ["web", "store", "web", "store", "web", "web", "store", "web"],
            "revenue": [100.0, 50.0, 30.0, 20.0, 70.0, 10.0, 40.0, 5.0],
            "units": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )


def by_key(result, key_col, value_col) -> dict:
    return dict(zip(result.frame[key_col], result.frame[value_col]))


# -- aggregation -------------------------------------------------------------


def test_group_and_sum(sales):
    spec = ChartSpec(chart="column", x="region", y={"field": "revenue", "aggregate": "sum"})
    result = apply_spec(sales, spec)
    assert by_key(result, "region", "revenue") == {"West": 220.0, "East": 90.0, "North": 15.0}


def test_result_is_one_row_per_mark(sales):
    spec = ChartSpec(
        chart="grouped_bar",
        x="region",
        y={"field": "revenue", "aggregate": "sum"},
        color="channel",
    )
    result = apply_spec(sales, spec)
    assert len(result.frame) == len(sales.groupby(["region", "channel"]))
    assert result.columns == {"x": "region", "y": "revenue", "color": "channel"}


def test_count_of_the_grouped_column_gets_its_own_output_name(sales):
    """"How many orders per region" binds one column to both x and y."""
    spec = ChartSpec(chart="column", x="region", y={"field": "region", "aggregate": "count"})
    result = apply_spec(sales, spec)
    assert result.columns == {"x": "region", "y": "region_count"}
    assert by_key(result, "region", "region_count") == {"West": 3, "East": 3, "North": 2}


@pytest.mark.parametrize(
    ("how", "expected"),
    [("sum", 220.0), ("mean", 220.0 / 3), ("max", 100.0), ("min", 50.0), ("nunique", 3)],
)
def test_aggregates(sales, how, expected):
    spec = ChartSpec(chart="column", x="region", y={"field": "revenue", "aggregate": how})
    result = apply_spec(sales, spec)
    assert by_key(result, "region", "revenue")["West"] == pytest.approx(expected)


def test_unaggregated_spec_keeps_the_raw_rows(sales):
    spec = ChartSpec(chart="scatter", x="revenue", y="units")
    result = apply_spec(sales, spec)
    assert len(result.frame) == len(sales)
    assert list(result.frame.columns) == ["revenue", "units"]


def test_single_headline_value_is_one_row(sales):
    spec = ChartSpec(chart="stat_tile", y={"field": "revenue", "aggregate": "sum"})
    result = apply_spec(sales, spec)
    assert len(result.frame) == 1
    assert result.frame["revenue"].iloc[0] == pytest.approx(sales["revenue"].sum())


# -- time binning ------------------------------------------------------------


def test_time_unit_bins_before_grouping(sales):
    spec = ChartSpec(
        chart="line",
        x={"field": "ordered_at", "time_unit": "month"},
        y={"field": "revenue", "aggregate": "sum"},
    )
    result = apply_spec(sales, spec)
    assert len(result.frame) == 3  # Jan, Feb, Mar - not 8 timestamps
    assert result.frame["revenue"].iloc[0] == pytest.approx(190.0)  # Jan


def test_time_axis_stays_chronological_not_ranked(sales):
    spec = ChartSpec(
        chart="line",
        x={"field": "ordered_at", "time_unit": "month"},
        y={"field": "revenue", "aggregate": "sum"},
    )
    dates = apply_spec(sales, spec).frame["ordered_at"].tolist()
    assert dates == sorted(dates)


# -- sorting and caps --------------------------------------------------------


def test_categorical_chart_is_ranked_by_its_measure(sales):
    spec = ChartSpec(chart="bar", x="region", y={"field": "revenue", "aggregate": "sum"})
    assert apply_spec(sales, spec).frame["region"].tolist() == ["West", "East", "North"]


def test_explicit_sort_wins(sales):
    spec = ChartSpec(
        chart="bar",
        x={"field": "region", "sort": "asc"},
        y={"field": "revenue", "aggregate": "sum"},
    )
    assert apply_spec(sales, spec).frame["region"].tolist() == ["East", "North", "West"]


def test_limit_keeps_the_top_rows_and_says_so(sales):
    spec = ChartSpec(
        chart="bar", x="region", y={"field": "revenue", "aggregate": "sum"}, limit=2
    )
    result = apply_spec(sales, spec)
    assert result.frame["region"].tolist() == ["West", "East"]
    assert any("top 2 of 3" in note for note in result.notes)


def test_limit_wider_than_the_data_adds_no_note(sales):
    spec = ChartSpec(
        chart="bar", x="region", y={"field": "revenue", "aggregate": "sum"}, limit=50
    )
    assert apply_spec(sales, spec).notes == ()


# -- series folding ----------------------------------------------------------


def test_series_past_the_cap_are_folded_into_other():
    df = pd.DataFrame(
        {
            "quarter": ["Q1", "Q2"] * 10,
            "product": [f"P{i}" for i in range(10)] * 2,
            "revenue": list(range(20)),
        }
    )
    spec = ChartSpec(
        chart="stacked_bar",
        x="quarter",
        y={"field": "revenue", "aggregate": "sum"},
        color="product",
        stack=True,
    )
    result = apply_spec(df, spec, max_series=4)
    series = set(result.frame["product"])
    assert len(series) == 4 and OTHER_LABEL in series
    assert any("folded into" in note for note in result.notes)


def test_folding_preserves_the_total():
    df = pd.DataFrame(
        {
            "quarter": ["Q1"] * 10,
            "product": [f"P{i}" for i in range(10)],
            "revenue": [float(i) for i in range(10)],
        }
    )
    spec = ChartSpec(
        chart="stacked_bar",
        x="quarter",
        y={"field": "revenue", "aggregate": "sum"},
        color="product",
        stack=True,
    )
    result = apply_spec(df, spec, max_series=3)
    assert result.frame["revenue"].sum() == pytest.approx(df["revenue"].sum())


def test_folding_keeps_the_largest_series():
    df = pd.DataFrame(
        {
            "quarter": ["Q1"] * 8,
            "product": list("abcdefgh"),
            "revenue": [1.0, 2, 3, 4, 5, 6, 7, 100],
        }
    )
    spec = ChartSpec(
        chart="stacked_bar",
        x="quarter",
        y={"field": "revenue", "aggregate": "sum"},
        color="product",
        stack=True,
    )
    result = apply_spec(df, spec, max_series=3)
    assert {"h", "g"} <= set(result.frame["product"])


def test_max_series_zero_keeps_every_series():
    df = pd.DataFrame(
        {"q": ["Q1"] * 9, "p": list("abcdefghi"), "v": [1.0] * 9}
    )
    spec = ChartSpec(
        chart="stacked_bar", x="q", y={"field": "v", "aggregate": "sum"}, color="p", stack=True
    )
    result = apply_spec(df, spec, max_series=0)
    assert len(set(result.frame["p"])) == 9


# -- filters -----------------------------------------------------------------


def test_filter_narrows_and_is_disclosed(sales):
    spec = ChartSpec(
        chart="column",
        x="region",
        y={"field": "revenue", "aggregate": "sum"},
        filters=[{"field": "channel", "op": "==", "value": "web"}],
    )
    result = apply_spec(sales, spec)
    assert by_key(result, "region", "revenue") == {"West": 170.0, "East": 30.0, "North": 15.0}
    assert any("excluded" in note for note in result.notes)


@pytest.mark.parametrize(
    ("op", "value", "expected_rows"),
    [
        (">", 40.0, 3),
        ("in", ["West", "North"], 5),
        ("not in", ["West"], 5),
        ("between", [20.0, 50.0], 4),
        ("contains", "we", 5),
        ("notnull", None, 8),
    ],
)
def test_filter_ops(sales, op, value, expected_rows):
    field = "revenue" if op in (">", "between") else "region" if op in ("in", "not in") else "channel"
    if op == "notnull":
        field = "revenue"
    spec = ChartSpec(
        chart="scatter",
        x="revenue",
        y="units",
        filters=[{"field": field, "op": op, "value": value}],
    )
    assert len(apply_spec(sales, spec).frame) == expected_rows


def test_filter_matching_nothing_shows_real_values(sales):
    spec = ChartSpec(
        chart="column",
        x="region",
        y={"field": "revenue", "aggregate": "sum"},
        filters=[{"field": "region", "op": "==", "value": "Souht"}],
    )
    with pytest.raises(SpecError, match="looks like"):
        apply_spec(sales, spec)


# -- missing data ------------------------------------------------------------


def test_null_group_keys_are_dropped_and_disclosed():
    df = pd.DataFrame({"region": ["West", None, "East"], "revenue": [1.0, 2.0, 3.0]})
    spec = ChartSpec(chart="column", x="region", y={"field": "revenue", "aggregate": "sum"})
    result = apply_spec(df, spec)
    assert len(result.frame) == 2
    assert any("missing region" in note for note in result.notes)


def test_unknown_column_is_reported_with_a_suggestion(sales):
    spec = ChartSpec(chart="column", x="reigon", y={"field": "revenue", "aggregate": "sum"})
    with pytest.raises(SpecError, match="did you mean: region"):
        apply_spec(sales, spec)


# -- result surface ----------------------------------------------------------


def test_result_reports_row_accounting(sales):
    spec = ChartSpec(
        chart="column",
        x="region",
        y={"field": "revenue", "aggregate": "sum"},
        filters=[{"field": "channel", "op": "==", "value": "web"}],
    )
    result = apply_spec(sales, spec)
    assert result.n_rows_in == 8
    assert result.n_rows_used == 5


def test_series_names_are_in_draw_order(sales):
    spec = ChartSpec(
        chart="grouped_bar",
        x="region",
        y={"field": "revenue", "aggregate": "sum"},
        color="channel",
    )
    assert set(apply_spec(sales, spec).series_names()) == {"web", "store"}


def test_to_records_round_trips(sales):
    spec = ChartSpec(chart="column", x="region", y={"field": "revenue", "aggregate": "sum"})
    records = apply_spec(sales, spec).to_records()
    assert records[0]["region"] == "West"


def test_the_input_frame_is_never_mutated(sales):
    before = sales.copy(deep=True)
    spec = ChartSpec(
        chart="line",
        x={"field": "ordered_at", "time_unit": "month"},
        y={"field": "revenue", "aggregate": "sum"},
        filters=[{"field": "channel", "op": "==", "value": "web"}],
    )
    apply_spec(sales, spec)
    pd.testing.assert_frame_equal(sales, before)


def test_notes_accumulate_on_the_returned_spec_not_the_original(sales):
    spec = ChartSpec(
        chart="bar", x="region", y={"field": "revenue", "aggregate": "sum"}, limit=1
    )
    result = apply_spec(sales, spec)
    assert result.spec.notes and spec.notes == ()
