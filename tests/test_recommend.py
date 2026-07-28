"""Tests for form choice.

These encode the form heuristic itself, so they read as claims about charts
rather than about code: a single number is not a bar chart, a trend is not
sorted by value, and nine series is a table rather than nine hues.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dataviz_agent.errors import SpecError
from dataviz_agent.profile import profile_frame
from dataviz_agent.recommend import MAX_MEANINGFUL_CLASSES, recommend


def prof(**columns) -> "object":
    return profile_frame(pd.DataFrame(columns))


@pytest.fixture
def sales():
    return prof(
        ordered_at=pd.date_range("2024-01-01", periods=24, freq="D"),
        region=["West", "East", "North", "South"] * 6,
        channel=["web", "store"] * 12,
        revenue=[float(i * 10) for i in range(24)],
    )


# -- is it even a chart ------------------------------------------------------


def test_a_single_aggregate_is_a_stat_tile_not_a_one_bar_chart(sales):
    rec = recommend(sales, y="revenue", intent="headline")
    assert rec.chart == "stat_tile"
    assert "not a chart" in rec.rationale


def test_a_lone_measure_with_nothing_to_compare_it_across_is_a_stat_tile():
    rec = recommend(prof(revenue=[1.0, 2.0, 3.0]))
    assert rec.chart == "stat_tile"


def test_a_measure_alongside_a_date_gets_the_time_axis(sales):
    """Naming only y is not the same as asking for a headline number."""
    assert recommend(sales, y="revenue").chart == "line"


def test_too_many_meaningful_classes_becomes_a_table():
    profile = prof(
        item=[f"item-{i}" for i in range(60)],
        bucket=[f"b{i % 60}" for i in range(60)],
        revenue=[float(i) for i in range(60)],
    )
    rec = recommend(profile, x="item", y="revenue", color="bucket")
    assert rec.chart == "table"
    assert "table" in rec.rationale


def test_series_past_the_ceiling_are_flagged_not_recolored():
    profile = prof(
        region=["a", "b"] * 10,
        product=[f"p{i % 9}" for i in range(20)],
        revenue=[float(i) for i in range(20)],
    )
    rec = recommend(profile, x="region", y="revenue", color="product")
    assert rec.chart in ("grouped_bar", "stacked_bar")
    assert any("Other" in note for note in rec.spec.notes)


# -- the job -> the form -----------------------------------------------------


def test_time_on_x_is_a_line(sales):
    rec = recommend(sales, x="ordered_at", y="revenue")
    assert rec.chart == "line"


def test_time_with_series_stays_a_line(sales):
    rec = recommend(sales, x="ordered_at", y="revenue", color="region")
    assert rec.chart == "line"
    assert rec.spec.color is not None


def test_composition_over_time_is_an_area(sales):
    rec = recommend(sales, x="ordered_at", y="revenue", color="region", intent="part_to_whole")
    assert rec.chart == "area"
    assert rec.spec.stack


def test_category_and_measure_is_a_bar_family_form(sales):
    assert recommend(sales, x="region", y="revenue").chart in ("bar", "column")


def test_two_measures_is_a_scatter(sales):
    profile = prof(spend=[1.0, 2, 3, 4], revenue=[2.0, 4, 6, 9])
    assert recommend(profile, x="spend", y="revenue").chart == "scatter"


def test_a_third_measure_makes_it_a_bubble():
    profile = prof(spend=[1.0, 2, 3], revenue=[2.0, 4, 6], reach=[10.0, 20, 30])
    rec = recommend(profile, x="spend", y="revenue", size="reach")
    assert rec.chart == "bubble"
    assert "scatter" in dict(rec.alternatives)


def test_one_measure_alone_is_a_distribution():
    profile = prof(latency_ms=[float(i) for i in range(100)])
    assert recommend(profile, x="latency_ms").chart == "histogram"


def test_two_categories_and_a_measure_is_a_grouped_bar(sales):
    rec = recommend(sales, x="region", y="revenue", color="channel")
    assert rec.chart == "grouped_bar"


def test_part_to_whole_across_categories_is_stacked(sales):
    rec = recommend(sales, x="region", y="revenue", color="channel", intent="part_to_whole")
    assert rec.chart == "stacked_bar"
    assert rec.spec.stack


def test_a_dense_category_grid_is_a_heatmap():
    profile = prof(
        hour=[f"h{i % 24}" for i in range(240)],
        weekday=[f"d{i % 8}" for i in range(240)],
        hits=[float(i) for i in range(240)],
    )
    rec = recommend(profile, x="hour", y="hits", color="weekday")
    assert rec.chart == "heatmap"


def test_negative_values_get_a_diverging_bar():
    profile = prof(region=list("abcd"), delta=[-5.0, 3.0, -2.0, 8.0])
    rec = recommend(profile, x="region", y="delta")
    assert rec.chart == "diverging_bar"


def test_emphasis_beats_a_categorical_palette(sales):
    rec = recommend(sales, x="region", y="revenue", emphasis="West")
    assert rec.spec.emphasis == "West"
    assert rec.spec.color_job == "emphasis"
    assert "context" in rec.rationale


def test_a_table_with_no_measure_counts_rows():
    profile = prof(status=["open", "closed", "open", "won"] * 5)
    rec = recommend(profile, x="status")
    assert rec.chart in ("bar", "column")
    assert rec.spec.y is not None and rec.spec.y.aggregate == "count"


# -- orientation and caps ----------------------------------------------------


def test_long_category_labels_go_horizontal():
    profile = prof(
        department=["Customer Success Operations", "Engineering Platform"] * 5,
        headcount=[float(i) for i in range(10)],
    )
    rec = recommend(profile, x="department", y="headcount")
    assert rec.chart == "bar"
    assert rec.spec.is_horizontal


def test_short_labels_stay_vertical():
    profile = prof(q=["Q1", "Q2", "Q3", "Q4"] * 3, revenue=[float(i) for i in range(12)])
    rec = recommend(profile, x="q", y="revenue")
    assert rec.chart == "column"
    assert not rec.spec.is_horizontal


def test_high_cardinality_axis_gets_a_top_n():
    profile = prof(
        sku=[f"sku-{i}" for i in range(120)], revenue=[float(i) for i in range(120)]
    )
    rec = recommend(profile, x="sku", y="revenue")
    assert rec.spec.limit is not None
    assert any("top" in note for note in rec.spec.notes)


def test_an_explicit_limit_is_respected(sales):
    assert recommend(sales, x="region", y="revenue", limit=3).spec.limit == 3


def test_dense_timestamps_are_binned():
    profile = prof(
        ts=pd.date_range("2024-01-01", periods=400, freq="D"),
        hits=[float(i) for i in range(400)],
    )
    rec = recommend(profile, x="ts", y="hits")
    assert rec.spec.x is not None and rec.spec.x.time_unit == "month"


def test_sparse_dates_are_left_alone(sales):
    assert recommend(sales, x="ordered_at", y="revenue").spec.x.time_unit is None


# -- automatic field choice --------------------------------------------------


def test_recommend_with_no_channels_picks_something_defensible(sales):
    rec = recommend(sales)
    assert rec.spec.x is not None and rec.spec.y is not None
    assert rec.spec.y.field == "revenue"
    assert rec.spec.x.field == "ordered_at"  # time wins as the axis


def test_a_metric_sounding_measure_is_preferred():
    profile = prof(
        region=["a", "b"] * 5,
        row_version=[float(i) for i in range(10)],
        revenue=[float(i) for i in range(10)],
    )
    assert recommend(profile).spec.y.field == "revenue"


def test_nothing_chartable_is_a_clear_error():
    profile = prof(note=[f"a long free text comment number {i}" * 4 for i in range(30)])
    with pytest.raises(SpecError, match="nothing chartable"):
        recommend(profile)


# -- overrides ---------------------------------------------------------------


def test_an_explicit_chart_is_honored(sales):
    assert recommend(sales, x="region", y="revenue", chart="stacked_bar", color="channel").chart == "stacked_bar"


def test_a_discouraged_form_is_honored_with_the_reason(sales):
    rec = recommend(sales, x="region", y="revenue", chart="pie")
    assert rec.chart == "pie"
    assert any("rarely the right form" in note for note in rec.spec.notes)


def test_pie_is_never_recommended_on_its_own(sales):
    for intent in ("part_to_whole", "magnitude", "comparison"):
        rec = recommend(sales, x="region", y="revenue", color="channel", intent=intent)
        assert rec.chart not in ("pie", "donut")


def test_unknown_intent_and_chart_are_rejected(sales):
    with pytest.raises(SpecError, match="unknown intent"):
        recommend(sales, x="region", y="revenue", intent="vibes")
    with pytest.raises(SpecError, match="unknown chart"):
        recommend(sales, x="region", y="revenue", chart="piechart")


# -- the recommendation surface ----------------------------------------------


def test_every_recommendation_explains_itself(sales):
    rec = recommend(sales, x="region", y="revenue")
    assert len(rec.rationale) > 30
    assert rec.explain().startswith(rec.chart)


def test_the_recommended_spec_is_valid_and_round_trips(sales):
    from dataviz_agent.spec import ChartSpec

    for kwargs in (
        {"x": "region", "y": "revenue"},
        {"x": "ordered_at", "y": "revenue", "color": "region"},
        {"x": "region", "y": "revenue", "color": "channel", "intent": "part_to_whole"},
        {"y": "revenue"},
    ):
        spec = recommend(sales, **kwargs).spec
        assert ChartSpec.from_json(spec.to_json()) == spec


def test_the_recommended_spec_executes(sales):
    """The recommender must not emit a spec the transform layer rejects."""
    from dataviz_agent.transform import apply_spec

    df = pd.DataFrame(
        {
            "ordered_at": pd.date_range("2024-01-01", periods=24, freq="D"),
            "region": ["West", "East", "North", "South"] * 6,
            "channel": ["web", "store"] * 12,
            "revenue": [float(i * 10) for i in range(24)],
        }
    )
    profile = profile_frame(df)
    for kwargs in (
        {},
        {"x": "region", "y": "revenue"},
        {"x": "ordered_at", "y": "revenue", "color": "region"},
        {"x": "region", "y": "revenue", "color": "channel", "intent": "part_to_whole"},
        {"x": "region", "y": "revenue", "emphasis": "West"},
        {"y": "revenue"},
        {"x": "revenue"},
    ):
        spec = recommend(profile, **kwargs).spec
        result = apply_spec(df, spec, profile=profile)
        assert len(result.frame) > 0
