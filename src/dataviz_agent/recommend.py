"""Choosing the form - from the data's job, not from a preference.

The rule this module encodes: **the job the reader has to do picks the chart,
and color comes last.** Magnitude across categories is a bar whether or not
someone asked for a pie; identity between series is the only thing that earns
a categorical palette; a single headline number is a stat tile, not a
one-bar bar chart.

Every recommendation carries a rationale and the runners-up. An agent that
disagrees can override the form - :mod:`dataviz_agent.spec` will still validate
it - but it has to do so knowingly, which is the point.

Two forms are supported and never recommended: pie and donut. Angle is read
less accurately than length, so they are honored on request and explained
rather than offered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .errors import SpecError
from .profile import MAX_CATEGORY_CARDINALITY, ColumnProfile, DataProfile
from .spec import CHART_TYPES, ChartSpec

__all__ = ["Intent", "Recommendation", "recommend", "INTENTS"]

Intent = Literal[
    "magnitude",  # compare sizes across categories
    "trend",  # change over time
    "part_to_whole",  # composition within a total
    "relationship",  # how two measures move together
    "distribution",  # the shape of one measure
    "comparison",  # tell distinct series apart
    "ranking",  # order by size, top-N
    "deviation",  # departure from a baseline
    "headline",  # a single number, or a handful
]
INTENTS: tuple[str, ...] = (
    "magnitude",
    "trend",
    "part_to_whole",
    "relationship",
    "distribution",
    "comparison",
    "ranking",
    "deviation",
    "headline",
)

#: Past this many classes a chart stops being the honest form and a table
#: starts. Straight from the series-count ladder: a 9th hue is indistinguishable
#: from an existing one under a colorblind simulation, so more colors is never
#: the fix for more series.
MAX_MEANINGFUL_CLASSES = 7

#: Categories beyond this get a top-N rather than one band each.
DEFAULT_TOP_N = 12

#: Category labels longer than this read better on a horizontal bar, where the
#: label has the full width of the margin instead of a tick's worth.
LONG_LABEL_CHARS = 12


@dataclass(frozen=True)
class Recommendation:
    """A chosen form, why it was chosen, and what came second."""

    spec: ChartSpec
    rationale: str
    alternatives: tuple[tuple[str, str], ...] = ()
    """``(chart, why you might prefer it)`` pairs, best first."""

    @property
    def chart(self) -> str:
        return self.spec.chart

    def explain(self) -> str:
        lines = [f"{self.chart}: {self.rationale}"]
        lines += [f"  or {name} - {why}" for name, why in self.alternatives]
        lines += [f"  note: {note}" for note in self.spec.notes]
        return "\n".join(lines)


def recommend(
    profile: DataProfile,
    *,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    size: str | None = None,
    facet: str | None = None,
    aggregate: str = "sum",
    intent: str | None = None,
    emphasis: str | None = None,
    time_unit: str | None = None,
    limit: int | None = None,
    mode: str = "light",
    query: str | None = None,
    chart: str | None = None,
) -> Recommendation:
    """Recommend a chart form for the given fields.

    Any channel left as ``None`` is chosen from the profile - so
    ``recommend(profile)`` alone produces the most defensible chart the table
    supports, and a caller who already knows what it wants on x and y just gets
    the form decision.

    Parameters
    ----------
    intent:
        A hint about the reader's job. Inferred from the column roles when
        omitted; supplying it mostly matters for distinguishing "compare these
        series" from "show what makes up the total", which the data shape alone
        cannot settle.
    chart:
        Force a form. Still validated against the data, and still annotated -
        asking for a discouraged form gets you the form plus the reason it is
        discouraged, not an argument.
    """
    if intent is not None and intent not in INTENTS:
        raise SpecError(f"unknown intent {intent!r}; expected one of {', '.join(INTENTS)}")
    if chart is not None and chart not in CHART_TYPES:
        raise SpecError(f"unknown chart type {chart!r}; supported: {', '.join(sorted(CHART_TYPES))}")

    cols = _resolve(profile, x=x, y=y, color=color, size=size, facet=facet)
    x_col, y_col, color_col, size_col, facet_col = (
        cols["x"], cols["y"], cols["color"], cols["size"], cols["facet"]
    )

    if y_col is None:
        y_col = _pick_measure(profile)
    if x_col is None and y_col is not None and intent != "headline":
        # "headline" is the caller saying the number itself is the story, so
        # nothing gets auto-filled onto x to compare it across.
        x_col = _pick_axis(profile, exclude={y_col.name})
    if x_col is None and y_col is None:
        raise SpecError(
            "nothing chartable in this table: no measure and no low-cardinality "
            f"dimension among {', '.join(profile.names)}"
        )

    # A table with no measure still has a chartable question: how many rows
    # fall in each category. Count the grouping column rather than refusing.
    if y_col is None and x_col is not None and not x_col.is_measure:
        y_col, aggregate = x_col, "count"

    intent = intent or _infer_intent(x_col, y_col, color_col, size_col, emphasis)
    notes: list[str] = []

    if chart is not None:
        form, rationale, alternatives = chart, "requested explicitly", ()
        meta = CHART_TYPES[chart]
        if meta.discouraged:
            notes.append(f"{chart} is rarely the right form: {meta.discouraged}")
    else:
        form, rationale, alternatives, extra_notes = _choose(
            profile, x_col, y_col, color_col, size_col, intent, emphasis
        )
        notes.extend(extra_notes)

    meta = CHART_TYPES[form]
    channels: dict[str, Any] = {}
    if "x" in meta.required or ("x" in meta.optional and x_col is not None):
        if x_col is not None:
            channels["x"] = _encode(x_col, time_unit=time_unit, form=form)
    if y_col is not None and ("y" in meta.required or "y" in meta.optional):
        channels["y"] = _encode_measure(y_col, aggregate, form=form, x_col=x_col)
    if color_col is not None and ("color" in meta.required or "color" in meta.optional):
        channels["color"] = color_col.name
    elif "color" in meta.required:
        raise SpecError(
            f"{form} needs a color channel but none was supplied or inferable; "
            "name the column whose values separate the series"
        )
    if size_col is not None and ("size" in meta.required or "size" in meta.optional):
        channels["size"] = _encode_measure(size_col, aggregate, form=form, x_col=x_col)
    if facet_col is not None and ("facet" in meta.optional or "facet" in meta.required):
        channels["facet"] = facet_col.name

    limit, notes = _resolve_limit(limit, form, x_col, color_col, notes)

    spec = ChartSpec(
        chart=form,
        **channels,
        emphasis=emphasis,
        limit=limit,
        stack=True if form in ("stacked_bar", "area") and color_col is not None else None,
        mode=mode,  # type: ignore[arg-type]
        query=query,
        notes=tuple(notes),
    )
    return Recommendation(spec=spec, rationale=rationale, alternatives=tuple(alternatives))


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------


def _choose(
    profile: DataProfile,
    x_col: ColumnProfile | None,
    y_col: ColumnProfile | None,
    color_col: ColumnProfile | None,
    size_col: ColumnProfile | None,
    intent: str,
    emphasis: str | None,
) -> tuple[str, str, list[tuple[str, str]], list[str]]:
    notes: list[str] = []
    alts: list[tuple[str, str]] = []

    # Is it even a chart? A single value has no shape to read.
    if x_col is None and y_col is not None:
        return (
            "stat_tile",
            "one aggregate with nothing to compare it across - a number, not a "
            "chart; a one-bar bar chart just adds an axis to read it against",
            [("meter", "if the number is a share of a known limit")],
            notes,
        )

    assert x_col is not None

    # Too many meaningful classes: a table, not more colors.
    if color_col is not None and color_col.n_unique > MAX_MEANINGFUL_CLASSES:
        if color_col.n_unique > MAX_CATEGORY_CARDINALITY:
            return (
                "table",
                f"{color_col.n_unique} distinct {color_col.name} values all carry "
                "meaning - past ~7 classes a table is the honest form, because a "
                "9th hue is indistinguishable from an existing one under a "
                "colorblind simulation",
                [("bar", "for a top-N of the same data")],
                notes,
            )
        notes.append(
            f"{color_col.n_unique} series is past the ~{MAX_MEANINGFUL_CLASSES}-class "
            "ceiling; the tail folds into 'Other' - facet or a table keeps them all"
        )

    # One measure, nothing to break it out by: a distribution. Checked before
    # the scatter branch, since "x and y are both measures" is trivially true
    # when they are the same column.
    if x_col.is_measure and (y_col is None or y_col is x_col):
        return (
            "histogram",
            "one continuous measure and nothing to compare it across - the shape "
            "of the distribution is the only story available",
            [("box", "to compare that distribution across a category")],
            notes,
        )

    # Two measures against each other.
    if x_col.is_measure and y_col is not None and y_col.is_measure:
        if size_col is not None:
            return (
                "bubble",
                "three measures at once - two on position, the third on mark area",
                [("scatter", "if the third measure is not essential; area is read "
                  "less precisely than position")],
                notes,
            )
        return (
            "scatter",
            "two continuous measures - position on both axes is the most precisely "
            "read encoding of a relationship",
            [("heatmap", "if the points overplot badly at this row count")],
            notes,
        )

    # Time on x.
    if x_col.role == "temporal":
        if color_col is not None:
            if intent == "part_to_whole":
                return (
                    "area",
                    "composition over time - stacked areas show the total and the "
                    "parts in one read",
                    [("line", "if the parts matter more than the total")],
                    notes,
                )
            return (
                "line",
                "trend over time with distinct series - lines keep each series "
                "readable where stacked forms would hide the individual shapes",
                [("area", "if the total is the point rather than each series")],
                notes,
            )
        return (
            "line",
            "a single measure over time - a line reads the trend directly",
            [
                ("area", "if the magnitude under the line is meaningful"),
                ("column", "if the time steps are few and discrete"),
            ],
            notes,
        )

    # Two dimensions and a measure: a grid.
    if color_col is not None and color_col.is_dimension and y_col.is_measure:
        # A heatmap earns its place on a real grid. With only a couple of x
        # bands it is a stacked bar wearing a ramp, and cells are read less
        # precisely than length.
        if (
            x_col.is_dimension
            and color_col.n_unique > MAX_MEANINGFUL_CLASSES
            and x_col.n_unique >= 5
        ):
            return (
                "heatmap",
                "magnitude over a two-dimensional grid of categories - too many "
                "cells for separate bars, and a single sequential ramp reads "
                "magnitude better than many hues",
                [("table", "if exact values matter more than the pattern")],
                notes,
            )
        if intent == "part_to_whole":
            return (
                "stacked_bar",
                "part-to-whole within each category - stacked lengths sum to the "
                "total while staying comparable",
                [("grouped_bar", "if the parts are compared to each other rather "
                  "than summed")],
                notes,
            )
        return (
            "grouped_bar",
            "distinct series compared within each category - side-by-side bars "
            "keep every series on a common baseline",
            [
                ("stacked_bar", "if the total across series is the point"),
                ("line", "if the x axis is ordered rather than nominal"),
            ],
            notes,
        )

    # One dimension, one measure.
    if emphasis is not None:
        alts.append(("grouped_bar", "if every series matters equally"))
        return (
            _bar_or_column(x_col),
            f"one category ({emphasis}) is the point and the rest are context - "
            "the accent hue on one bar with the others in the de-emphasis gray "
            "says that directly, where a categorical palette would bury it",
            alts,
            notes,
        )

    if intent == "deviation" or (y_col.has_negative and intent in ("magnitude", "ranking")):
        return (
            "diverging_bar",
            "values fall on both sides of a baseline - a diverging ramp with a "
            "neutral midpoint makes the sign readable without reading the axis",
            [("bar", "if the sign is incidental rather than the story")],
            notes,
        )

    if not x_col.is_measure and not x_col.is_drawable_axis:
        notes.append(
            f"{x_col.n_unique} distinct {x_col.name} values - showing the top "
            f"{DEFAULT_TOP_N}"
        )

    form = _bar_or_column(x_col)
    return (
        form,
        "magnitude across discrete categories - length from a common baseline is "
        "the most accurately read comparison, on a single-hue sequential ramp "
        "because the categories are the axis, not the subject",
        [
            ("stat_tile", "if one of these numbers is the actual headline"),
            ("table", "if the exact values matter more than the comparison"),
        ],
        notes,
    )


def _bar_or_column(x_col: ColumnProfile) -> str:
    """Horizontal for long or numerous labels, vertical otherwise."""
    labels = [str(v) for v, _ in x_col.top_values] or [str(v) for v in x_col.sample]
    longest = max((len(s) for s in labels), default=0)
    if longest > LONG_LABEL_CHARS or x_col.n_unique > 8:
        return "bar"
    return "column"


def _infer_intent(
    x_col: ColumnProfile | None,
    y_col: ColumnProfile | None,
    color_col: ColumnProfile | None,
    size_col: ColumnProfile | None,
    emphasis: str | None,
) -> str:
    if x_col is None:
        return "headline"
    if emphasis is not None:
        return "comparison"
    if x_col.role == "temporal":
        return "trend"
    if x_col.is_measure and (y_col is None or y_col.is_measure):
        return "relationship" if y_col is not None else "distribution"
    if color_col is not None:
        return "comparison"
    if y_col is not None and y_col.has_negative:
        return "deviation"
    return "magnitude"


# --------------------------------------------------------------------------
# Field selection
# --------------------------------------------------------------------------


def _resolve(profile: DataProfile, **channels: str | None) -> dict[str, ColumnProfile | None]:
    return {
        channel: (profile.resolve(name) if name is not None else None)
        for channel, name in channels.items()
    }


def _pick_measure(profile: DataProfile) -> ColumnProfile | None:
    """The most chart-worthy measure: the one with the fewest gaps, preferring
    a name that reads like an amount over an incidental count."""
    measures = profile.measures
    if not measures:
        return None
    return sorted(
        measures,
        key=lambda c: (c.missing_ratio, not _sounds_like_a_metric(c.name), profile.names.index(c.name)),
    )[0]


def _pick_axis(profile: DataProfile, *, exclude: set[str]) -> ColumnProfile | None:
    """Time if there is any; otherwise the smallest drawable dimension."""
    candidates = [c for c in profile.dimensions if c.name not in exclude and c.is_drawable_axis]
    if not candidates:
        # Nothing small enough to draw as bands - fall back to any dimension so
        # the top-N path can take over rather than failing outright.
        candidates = [c for c in profile.dimensions if c.name not in exclude]
    return candidates[0] if candidates else None


_METRIC_WORDS = (
    "revenue", "sales", "amount", "total", "value", "price", "cost", "profit",
    "margin", "spend", "score", "rate", "duration", "time", "count", "qty",
    "quantity", "units", "views", "clicks", "users",
)


def _sounds_like_a_metric(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in _METRIC_WORDS)


# --------------------------------------------------------------------------
# Channel construction
# --------------------------------------------------------------------------


def _encode(col: ColumnProfile, *, time_unit: str | None, form: str) -> Any:
    if col.role == "temporal":
        unit = time_unit or _default_time_unit(col)
        if unit:
            return {"field": col.name, "time_unit": unit}
    return col.name


def _default_time_unit(col: ColumnProfile) -> str | None:
    """Bin dense timestamps so a trend has readable steps.

    Only when there are enough distinct values to be worth binning - a dozen
    month-end dates are already the right granularity.
    """
    return "month" if col.n_unique > 60 else None


def _encode_measure(
    col: ColumnProfile, aggregate: str, *, form: str, x_col: ColumnProfile | None
) -> Any:
    """Aggregate a measure only when the x channel groups rows together."""
    if form in ("scatter", "bubble", "histogram"):
        return col.name
    if x_col is not None and x_col.is_measure and form not in ("bar", "column"):
        return col.name
    return {"field": col.name, "aggregate": aggregate}


def _resolve_limit(
    limit: int | None,
    form: str,
    x_col: ColumnProfile | None,
    color_col: ColumnProfile | None,
    notes: list[str],
) -> tuple[int | None, list[str]]:
    if limit is not None or form == "table":
        return limit, notes
    if x_col is not None and not x_col.is_measure and not x_col.is_drawable_axis:
        return DEFAULT_TOP_N, notes
    return None, notes
