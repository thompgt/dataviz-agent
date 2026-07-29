"""Tests for the no-model query parser.

Coverage is not the goal - honesty is. The parser must get the common phrasings
right, report what it inferred, and refuse to guess between two equally good
readings.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dataviz_agent.errors import AmbiguousQueryError, SpecError
from dataviz_agent.nlq import parse_query
from dataviz_agent.profile import profile_frame


@pytest.fixture
def sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ordered_at": pd.date_range("2024-01-01", periods=40, freq="D"),
            "region": ["West", "East", "North", "South"] * 10,
            "channel": ["web", "store"] * 20,
            "revenue": [float(i * 10) for i in range(40)],
            "units": list(range(40)),
        }
    )


@pytest.fixture
def sales(sales_df):
    return profile_frame(sales_df)


def parse(sales, text):
    return parse_query(text, sales)


# -- the common phrasings ----------------------------------------------------


def test_total_measure_by_dimension(sales):
    parsed = parse(sales, "total revenue by region")
    assert parsed.spec.y.field == "revenue"
    assert parsed.spec.y.aggregate == "sum"
    assert parsed.spec.x.field == "region"
    assert parsed.chart in ("bar", "column")


def test_average_is_not_a_sum(sales):
    assert parse(sales, "average revenue by channel").spec.y.aggregate == "mean"


def test_over_time_becomes_a_trend(sales):
    parsed = parse(sales, "revenue over time")
    assert parsed.chart == "line"
    assert parsed.spec.x.field == "ordered_at"


def test_by_month_bins_time(sales):
    parsed = parse(sales, "total revenue by month")
    assert parsed.spec.x.field == "ordered_at"
    assert parsed.spec.x.time_unit == "month"


def test_two_groupings_become_axis_and_series(sales):
    parsed = parse(sales, "revenue by region and channel")
    assert parsed.spec.x.field == "region"
    assert parsed.spec.color.field == "channel"
    assert parsed.chart in ("grouped_bar", "stacked_bar")


def test_breakdown_wording_stacks(sales):
    parsed = parse(sales, "breakdown of revenue by region and channel")
    assert parsed.chart == "stacked_bar"


def test_counting_rows_needs_no_measure(sales):
    parsed = parse(sales, "how many orders by region")
    assert parsed.spec.y.aggregate == "count"
    assert parsed.spec.x.field == "region"


def test_two_measures_read_as_a_relationship(sales):
    parsed = parse(sales, "revenue vs units")
    assert parsed.chart == "scatter"
    assert {parsed.spec.x.field, parsed.spec.y.field} == {"revenue", "units"}


def test_distribution_wording(sales):
    assert parse(sales, "distribution of revenue").chart == "histogram"


# -- modifiers ---------------------------------------------------------------


def test_top_n_sets_a_limit(sales):
    parsed = parse(sales, "top 3 regions by revenue")
    assert parsed.spec.limit == 3


def test_bottom_n_sorts_the_other_way(sales):
    parsed = parse(sales, "bottom 2 regions by revenue")
    assert parsed.spec.limit == 2
    assert parsed.spec.x.sort == "asc"


def test_a_year_becomes_a_date_filter(sales):
    parsed = parse(sales, "revenue by region in 2024")
    assert parsed.spec.filters
    flt = parsed.spec.filters[0]
    assert flt.field == "ordered_at" and flt.op == "between"


def test_a_literal_category_value_becomes_a_filter(sales):
    parsed = parse(sales, "revenue by region for web")
    assert any(f.field == "channel" and f.value == "web" for f in parsed.spec.filters)


def test_an_explicit_chart_type_is_honored(sales):
    assert parse(sales, "revenue by region as a table").chart == "table"


def test_a_requested_pie_is_drawn_and_explained(sales):
    parsed = parse(sales, "pie chart of revenue by region")
    assert parsed.chart == "pie"
    assert any("not recommended" in line for line in parsed.interpretation)


# -- honesty -----------------------------------------------------------------


def test_the_interpretation_is_reported(sales):
    parsed = parse(sales, "average revenue by month")
    joined = " ".join(parsed.interpretation)
    assert "mean" in joined and "month" in joined


def test_ambiguous_column_reference_asks_instead_of_guessing():
    profile = profile_frame(
        pd.DataFrame(
            {
                "west_revenue": [1.0, 2.0, 3.0],
                "east_revenue": [1.0, 2.0, 3.0],
                "day": pd.date_range("2024-01-01", periods=3),
            }
        )
    )
    with pytest.raises(AmbiguousQueryError) as excinfo:
        parse_query("revenue over time", profile)
    assert set(excinfo.value.candidates) == {"west_revenue", "east_revenue"}


def test_a_question_about_columns_that_do_not_exist_fails_loudly(sales):
    with pytest.raises(AmbiguousQueryError, match="nothing in"):
        parse(sales, "show me the churn cohort waterfall")


def test_unmatched_words_are_reported(sales):
    parsed = parse(sales, "revenue by region excluding cancellations")
    assert "cancellations" in parsed.unmatched


def test_empty_query_is_rejected(sales):
    with pytest.raises(SpecError):
        parse(sales, "   ")


def test_explain_mentions_the_query_and_the_form(sales):
    text = parse(sales, "total revenue by region").explain()
    assert "total revenue by region" in text
    assert "=>" in text


# -- end to end --------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "total revenue by region",
        "average revenue by channel",
        "revenue over time",
        "total revenue by month",
        "revenue by region and channel",
        "top 3 regions by revenue",
        "how many orders by region",
        "revenue vs units",
        "distribution of revenue",
        "revenue by region in 2024",
        "breakdown of revenue by region and channel",
    ],
)
def test_every_parsed_question_executes(sales_df, sales, question):
    """A spec the parser emits must survive the transform layer."""
    from dataviz_agent.transform import apply_spec

    parsed = parse_query(question, sales)
    result = apply_spec(sales_df, parsed.spec, profile=sales)
    assert len(result.frame) > 0
