"""Plain English in, a ChartSpec out - with no model in the loop.

This is the deterministic half of query understanding. It handles the phrasing
that has one honest reading: "total revenue by region", "orders per month in
2024", "top 5 products", "revenue vs spend". An LLM agent driving the toolkit
can skip it entirely and emit a :class:`~dataviz_agent.spec.ChartSpec` directly
- but the package should not *require* a model to answer a question a regular
expression can answer.

Where it will not guess
-----------------------
Two behaviours matter more than coverage:

* When a word matches two columns equally well, it raises
  :class:`~dataviz_agent.errors.AmbiguousQueryError` carrying both candidates,
  instead of picking one. Silently charting the wrong column is worse than
  asking.
* Everything it inferred is reported in :attr:`ParsedQuery.interpretation`, so
  the caller can see that "sales" was read as ``net_rev_usd`` and that "last
  year" became a date filter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .errors import AmbiguousQueryError, SpecError
from .profile import ColumnProfile, DataProfile, _words
from .recommend import Recommendation, recommend
from .spec import CHART_TYPES, ChartSpec

__all__ = ["ParsedQuery", "parse_query"]


# -- vocabulary --------------------------------------------------------------

_AGGREGATE_WORDS: dict[str, tuple[str, ...]] = {
    "sum": ("total", "sum", "summed", "combined", "overall"),
    "mean": ("average", "avg", "mean", "typical", "per capita"),
    "median": ("median",),
    "max": ("max", "maximum", "highest", "peak", "largest", "best"),
    "min": ("min", "minimum", "lowest", "smallest", "worst"),
    "count": ("count", "number", "how many", "frequency", "tally", "volume"),
    "nunique": ("distinct", "unique", "different"),
    "std": ("spread", "deviation", "variance", "volatility"),
}

_INTENT_PHRASES: dict[str, tuple[str, ...]] = {
    "trend": ("over time", "trend", "trending", "growth", "history", "historical", "by month", "by year", "by day", "by week", "by quarter"),
    "part_to_whole": ("breakdown", "composition", "share of", "makes up", "made up", "proportion", "percentage of", "split of", "mix"),
    "relationship": ("vs", "versus", "against", "correlation", "correlate", "related to", "relationship"),
    "distribution": ("distribution", "histogram", "spread of", "how are", "range of"),
    "ranking": ("top", "bottom", "rank", "ranked", "ranking", "leaderboard", "best", "worst"),
    "comparison": ("compare", "comparison", "difference between", "by segment"),
    "deviation": ("change", "delta", "difference from", "above or below", "variance from", "vs target"),
    "headline": ("how much", "how many total", "what is the total", "overall total"),
}

_TIME_UNIT_WORDS: dict[str, tuple[str, ...]] = {
    "year": ("year", "yearly", "annual", "annually", "per year", "by year"),
    "quarter": ("quarter", "quarterly", "per quarter", "by quarter"),
    "month": ("month", "monthly", "per month", "by month"),
    "week": ("week", "weekly", "per week", "by week"),
    "day": ("day", "daily", "per day", "by day", "date"),
    "hour": ("hour", "hourly", "per hour", "by hour"),
}

_CHART_WORDS: dict[str, tuple[str, ...]] = {
    "pie": ("pie", "pie chart"),
    "donut": ("donut", "doughnut"),
    "line": ("line chart", "line graph", "as a line"),
    "area": ("area chart", "stacked area"),
    "scatter": ("scatter", "scatterplot", "scatter plot"),
    "bubble": ("bubble chart", "bubble"),
    "heatmap": ("heatmap", "heat map"),
    "histogram": ("histogram",),
    "box": ("box plot", "boxplot", "box and whisker"),
    "table": ("table", "as a table", "list"),
    "stacked_bar": ("stacked bar", "stacked column"),
    "grouped_bar": ("grouped bar", "clustered bar", "side by side"),
    "bar": ("bar chart", "bar graph"),
    "column": ("column chart",),
    "stat_tile": ("stat tile", "big number", "headline number", "kpi"),
    "dumbbell": ("dumbbell", "before and after"),
}

#: Words that are grammar, not data - never matched against column names.
_STOPWORDS = frozenset(
    """
    a an and are as at be by chart do does for from graph how in into is it me
    of on or over per plot show shows the their there this to us was were what
    when where which who with give please can you my our their s
    """.split()
)

_GROUP_MARKERS = ("by", "per", "across", "for each", "grouped by", "broken down by", "split by")


@dataclass(frozen=True)
class ParsedQuery:
    """A question, and everything that was concluded from it."""

    query: str
    recommendation: Recommendation
    interpretation: tuple[str, ...] = ()
    """Plain-language account of each inference, in the order they were made.
    Shown to the user, because a chart that answered a subtly different
    question is only catchable if the reading is visible."""

    unmatched: tuple[str, ...] = ()
    """Content words no column, value or keyword accounted for. A long list
    here is the signal that the question needed a model, not a parser."""

    @property
    def spec(self) -> ChartSpec:
        return self.recommendation.spec

    @property
    def chart(self) -> str:
        return self.recommendation.chart

    def explain(self) -> str:
        lines = [f"{self.query!r}", *(f"  - {line}" for line in self.interpretation)]
        lines.append(f"  => {self.recommendation.explain()}")
        if self.unmatched:
            lines.append(f"  (unused words: {', '.join(self.unmatched)})")
        return "\n".join(lines)


def parse_query(query: str, profile: DataProfile, *, mode: str = "light") -> ParsedQuery:
    """Parse *query* against *profile* and return a recommended chart.

    Raises
    ------
    AmbiguousQueryError
        When a term in the question matches more than one column equally well.
        The ``candidates`` attribute carries the competing readings so the
        caller can ask rather than guess.
    SpecError
        When nothing in the question maps onto the data at all.
    """
    if not isinstance(query, str) or not query.strip():
        raise SpecError("query must be a non-empty string")

    text = query.lower().strip()
    notes: list[str] = []
    used_spans: list[tuple[int, int]] = []

    chart = _find_chart(text, notes, used_spans)
    aggregate = _find_aggregate(text, notes, used_spans)
    time_unit = _find_time_unit(text, notes, used_spans)
    limit, ranking_dir = _find_limit(text, notes, used_spans)
    intent = _find_intent(text, notes, used_spans, limit is not None)

    matches = _match_columns(text, profile, used_spans)
    measure, dimensions = _assign_roles(text, profile, matches, notes)
    filters = _find_filters(text, profile, matches, measure, dimensions, notes, used_spans)

    if measure is None and not dimensions:
        raise AmbiguousQueryError(
            f"nothing in {query!r} names a column in this table "
            f"({', '.join(profile.names)}); rephrase using a column name, or "
            "emit a ChartSpec directly",
            candidates=list(profile.names),
        )

    x = dimensions[0].name if dimensions else None
    color = dimensions[1].name if len(dimensions) > 1 else None
    if x is not None and color is not None:
        notes.append(f"grouping by {x}, with {color} as the series")

    if measure is not None and aggregate is None:
        aggregate = "count" if measure.role != "quantitative" else "sum"
        notes.append(f"no aggregate named - totalling {measure.name}" if aggregate == "sum"
                     else f"no aggregate named - counting {measure.name}")

    if intent == "relationship" and measure is not None:
        # "revenue vs spend" - two measures, not a measure and a grouping.
        others = [c for c in matches.measures if c is not measure]
        if others:
            x, color = others[0].name, color
            notes.append(f"reading {others[0].name} against {measure.name} as a relationship")

    rec = recommend(
        profile,
        x=x,
        y=measure.name if measure is not None else None,
        color=color,
        aggregate=aggregate or "sum",
        intent=intent,
        time_unit=time_unit,
        limit=limit,
        chart=chart,
        mode=mode,
        query=query,
    )
    spec = rec.spec
    if filters:
        spec = spec.with_changes(filters=tuple(filters))
    if ranking_dir == "bottom" and spec.x is not None:
        spec = spec.with_changes(x=_sorted(spec.x, "asc"))
        notes.append("'bottom' - sorting ascending so the smallest values lead")
    rec = Recommendation(spec=spec, rationale=rec.rationale, alternatives=rec.alternatives)

    return ParsedQuery(
        query=query,
        recommendation=rec,
        interpretation=tuple(notes),
        unmatched=_leftover_words(text, used_spans),
    )


# --------------------------------------------------------------------------
# Keyword extraction
# --------------------------------------------------------------------------


def _find_phrase(text: str, phrases: Iterable[str]) -> tuple[str, int, int] | None:
    """Longest phrase match, so 'by quarter' beats 'by'."""
    best: tuple[str, int, int] | None = None
    for phrase in phrases:
        for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
            if best is None or len(phrase) > len(best[0]):
                best = (phrase, match.start(), match.end())
    return best


def _find_in_table(
    text: str, table: dict[str, tuple[str, ...]], used: list[tuple[int, int]]
) -> tuple[str, str] | None:
    """Best (key, phrase) across a whole vocabulary table, longest phrase wins."""
    best: tuple[str, str, int, int] | None = None
    for key, phrases in table.items():
        found = _find_phrase(text, phrases)
        if found and (best is None or len(found[0]) > len(best[1])):
            best = (key, found[0], found[1], found[2])
    if best is None:
        return None
    used.append((best[2], best[3]))
    return best[0], best[1]


def _find_chart(text: str, notes: list[str], used: list[tuple[int, int]]) -> str | None:
    found = _find_in_table(text, _CHART_WORDS, used)
    if found is None:
        return None
    chart, phrase = found
    notes.append(f"'{phrase}' - drawing a {chart}")
    if CHART_TYPES[chart].discouraged:
        notes.append(f"{chart} was requested, not recommended: {CHART_TYPES[chart].discouraged}")
    return chart


def _find_aggregate(text: str, notes: list[str], used: list[tuple[int, int]]) -> str | None:
    found = _find_in_table(text, _AGGREGATE_WORDS, used)
    if found is None:
        return None
    how, phrase = found
    # "revenue" is both a column word and a sum cue; only report the cue when
    # it is not simply the name of the measure.
    notes.append(f"'{phrase}' - aggregating with {how}")
    return how


def _find_time_unit(text: str, notes: list[str], used: list[tuple[int, int]]) -> str | None:
    found = _find_in_table(text, _TIME_UNIT_WORDS, used)
    if found is None:
        return None
    unit, phrase = found
    notes.append(f"'{phrase}' - binning time by {unit}")
    return unit


def _find_intent(
    text: str, notes: list[str], used: list[tuple[int, int]], has_limit: bool
) -> str | None:
    found = _find_in_table(text, _INTENT_PHRASES, used)
    if found is None:
        return "ranking" if has_limit else None
    intent, phrase = found
    notes.append(f"'{phrase}' - reading this as {intent.replace('_', '-')}")
    return intent


_LIMIT_RE = re.compile(r"\b(top|bottom|first|last)\s+(\d{1,4})\b")


def _find_limit(
    text: str, notes: list[str], used: list[tuple[int, int]]
) -> tuple[int | None, str | None]:
    match = _LIMIT_RE.search(text)
    if match is None:
        return None, None
    used.append((match.start(), match.end()))
    word, number = match.group(1), int(match.group(2))
    direction = "bottom" if word in ("bottom", "last") else "top"
    notes.append(f"'{match.group(0)}' - capping to {number} ({direction})")
    return number, direction


# --------------------------------------------------------------------------
# Column matching
# --------------------------------------------------------------------------


@dataclass
class _Matches:
    """Columns the question named, in the order they were named."""

    ordered: list[ColumnProfile] = field(default_factory=list)
    spans: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def measures(self) -> list[ColumnProfile]:
        return [c for c in self.ordered if c.is_measure]

    @property
    def dimensions(self) -> list[ColumnProfile]:
        return [c for c in self.ordered if c.is_dimension]


def _match_columns(
    text: str, profile: DataProfile, used: list[tuple[int, int]]
) -> _Matches:
    """Find every column the question names, by name or by synonym."""
    matches = _Matches()
    claimed: list[tuple[int, int]] = []

    candidates: list[tuple[int, int, ColumnProfile, str]] = []
    for col in profile.columns:
        for phrase in _column_phrases(col):
            for hit in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
                candidates.append((hit.start(), hit.end(), col, phrase))

    # Longest match first, so "order date" beats "order" and "date".
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
    for start, end, col, phrase in candidates:
        if any(start < e and s < end for s, e in claimed):
            continue
        rival = _rival_column(profile, col, phrase)
        if rival is not None:
            raise AmbiguousQueryError(
                f"{phrase!r} matches both {col.name!r} and {rival.name!r}; "
                "name the column you mean",
                candidates=[col.name, rival.name],
            )
        claimed.append((start, end))
        used.append((start, end))
        if col not in matches.ordered:
            matches.ordered.append(col)
            matches.spans[col.name] = (start, end)

    matches.ordered.sort(key=lambda c: matches.spans[c.name][0])
    return matches


def _column_phrases(col: ColumnProfile) -> list[str]:
    """Every spelling of a column name worth matching, longest first."""
    words = _words(col.name)
    spellings = {" ".join(words), "".join(words), col.name.lower()}
    if len(words) > 1:
        # 'ordered at' is a phrase nobody types; 'ordered' usually suffices.
        spellings.add(words[0])
        # The head noun is how people actually refer to a qualified column:
        # 'revenue' for 'west_revenue'. It is deliberately the loosest match
        # here - when two columns share one, _rival_column turns it into a
        # question rather than a coin flip.
        spellings.add(words[-1])
    return sorted((s for s in spellings if s and s not in _STOPWORDS), key=len, reverse=True)


def _rival_column(
    profile: DataProfile, col: ColumnProfile, phrase: str
) -> ColumnProfile | None:
    """Another column the same phrase names just as well."""
    for other in profile.columns:
        if other.name == col.name:
            continue
        if phrase in _column_phrases(other) and len(_words(other.name)) == len(_words(col.name)):
            return other
    return None


def _assign_roles(
    text: str,
    profile: DataProfile,
    matches: _Matches,
    notes: list[str],
) -> tuple[ColumnProfile | None, list[ColumnProfile]]:
    """Decide which named column is the measure and which group the rows.

    A "by"/"per" marker is the strongest signal available: whatever follows it
    is a grouping, whatever precedes it is the thing being measured.
    """
    grouped: list[ColumnProfile] = []
    marker = _find_phrase(text, _GROUP_MARKERS)
    if marker is not None:
        cut = marker[2]
        grouped = [
            c for c in matches.ordered
            if matches.spans[c.name][0] >= cut and not c.is_measure
        ]
        if grouped:
            notes.append(
                f"'{marker[0]}' - grouping by {', '.join(c.name for c in grouped)}"
            )

    measures = [c for c in matches.measures if c not in grouped]
    measure = measures[0] if measures else None

    dimensions = grouped or [c for c in matches.dimensions if c is not measure]
    if measure is None and dimensions:
        # "how many orders by region" - the thing counted is the grouping
        # itself, and count is the only aggregate that makes sense.
        measure = dimensions[0]
    return measure, dimensions


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(?:in|during|for)\s+(19|20)\d{2}\b")


def _find_filters(
    text: str,
    profile: DataProfile,
    matches: _Matches,
    measure: ColumnProfile | None,
    dimensions: Sequence[ColumnProfile],
    notes: list[str],
    used: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Recognise the two filter forms a question states unambiguously: a
    literal category value, and a calendar year."""
    filters: list[dict[str, Any]] = []

    year = _YEAR_RE.search(text)
    if year is not None:
        temporal = profile.temporal
        if temporal:
            value = year.group(0).split()[-1]
            used.append((year.start(), year.end()))
            filters.append(
                {
                    "field": temporal[0].name,
                    "op": "between",
                    "value": [f"{value}-01-01", f"{value}-12-31"],
                }
            )
            notes.append(f"'{year.group(0)}' - filtering {temporal[0].name} to {value}")

    for col in profile.columns:
        if col.role not in ("categorical", "boolean") or not col.top_values:
            continue
        for value, _count in col.top_values:
            token = str(value).lower()
            if len(token) < 3 or token in _STOPWORDS:
                continue
            hit = re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text)
            if hit is None or any(s <= hit.start() < e for s, e in used):
                continue
            used.append((hit.start(), hit.end()))
            filters.append({"field": col.name, "op": "==", "value": value})
            notes.append(f"'{token}' is a {col.name} value - filtering to it")
            break

    return filters


# --------------------------------------------------------------------------
# Odds and ends
# --------------------------------------------------------------------------


def _sorted(encoding: Any, direction: str) -> Any:
    from dataclasses import replace

    return replace(encoding, sort=direction)


def _leftover_words(text: str, used: Sequence[tuple[int, int]]) -> tuple[str, ...]:
    """Content words nothing accounted for - the signal that this question
    needed a model rather than a parser."""
    leftover: list[str] = []
    for match in re.finditer(r"[a-z_][a-z0-9_]*", text):
        if any(s < match.end() and match.start() < e for s, e in used):
            continue
        word = match.group(0)
        if word in _STOPWORDS or len(word) < 3:
            continue
        leftover.append(word)
    return tuple(dict.fromkeys(leftover))
