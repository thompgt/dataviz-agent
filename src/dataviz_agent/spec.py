"""The ChartSpec - a validated, serializable description of exactly one chart.

This module is the seam between the agent and the toolkit. An LLM agent can
emit a ChartSpec as JSON; :mod:`dataviz_agent.nlq` can derive one from plain
English; :mod:`dataviz_agent.transform` executes one against a DataFrame; and
:mod:`dataviz_agent.render` draws one. Nothing downstream re-interprets intent,
because by this point intent is data.

The chart registry
------------------
:data:`CHART_TYPES` is the single source of truth about chart forms. It drives
three things that would otherwise drift apart:

* **validation** - which encodings a form requires and permits,
* **the recommender** - which forms are candidates for a given data shape,
* **the renderer** - which color job and pairlist a form needs.

Adding a chart form means adding one registry entry and one draw function; it
should never mean touching an ``if chart_type == ...`` chain in four files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Literal, Mapping, Sequence

from .errors import SpecError, did_you_mean

__all__ = [
    "Aggregate",
    "ChartType",
    "ColorJob",
    "Encoding",
    "Filter",
    "ChartSpec",
    "ChartMeta",
    "CHART_TYPES",
    "AGGREGATES",
    "TIME_UNITS",
    "FILTER_OPS",
]

Aggregate = Literal[
    "sum", "mean", "median", "min", "max", "count", "nunique", "std", "none"
]
AGGREGATES: tuple[str, ...] = (
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
    "nunique",
    "std",
    "none",
)

TIME_UNITS: tuple[str, ...] = ("year", "quarter", "month", "week", "day", "hour")

FILTER_OPS: tuple[str, ...] = (
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "in",
    "not in",
    "contains",
    "between",
    "isnull",
    "notnull",
)

ColorJob = Literal["categorical", "sequential", "diverging", "status", "emphasis", "none"]
ChartType = str  # constrained at runtime by CHART_TYPES


@dataclass(frozen=True)
class ChartMeta:
    """Static facts about a chart form.

    Attributes
    ----------
    required:
        Encoding channels that must be present.
    optional:
        Channels the form understands. Anything outside ``required | optional``
        is rejected rather than ignored - silently dropping an encoding the
        caller asked for produces a chart they did not request.
    color_job:
        The default job color does in this form, which selects the ramp.
    pairs:
        ``"all"`` for forms where any two series can land adjacent (scatter,
        bubble, heatmap cells, small multiples); ``"adjacent"`` otherwise.
        This picks the palette pairlist and therefore the series cap.
    continuous_x:
        Whether the x channel is naturally continuous (time / number) rather
        than a set of discrete bands.
    discouraged:
        Populated when the form is supported but is rarely the right answer;
        the string explains what to reach for instead. The recommender never
        selects a discouraged form, but an explicit request is honored.
    """

    name: str
    summary: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    color_job: ColorJob = "categorical"
    pairs: Literal["adjacent", "all"] = "adjacent"
    continuous_x: bool = False
    discouraged: str | None = None


def _m(*args: Any, **kwargs: Any) -> ChartMeta:
    return ChartMeta(*args, **kwargs)


CHART_TYPES: dict[str, ChartMeta] = {
    "bar": _m(
        "bar",
        "Magnitude across discrete categories, drawn horizontally.",
        required=("x", "y"),
        optional=("color", "facet", "emphasis"),
        color_job="sequential",
    ),
    "column": _m(
        "column",
        "Magnitude across discrete categories, drawn vertically.",
        required=("x", "y"),
        optional=("color", "facet", "emphasis"),
        color_job="sequential",
    ),
    "grouped_bar": _m(
        "grouped_bar",
        "Compare a few distinct series side by side within each category.",
        required=("x", "y", "color"),
        optional=("facet",),
        color_job="categorical",
    ),
    "stacked_bar": _m(
        "stacked_bar",
        "Part-to-whole within each category. Go horizontal for long category names.",
        required=("x", "y", "color"),
        optional=("facet",),
        color_job="categorical",
    ),
    "diverging_bar": _m(
        "diverging_bar",
        "Departure from a baseline - above/below zero, or delta to target.",
        required=("x", "y"),
        optional=("color", "facet"),
        color_job="diverging",
    ),
    "line": _m(
        "line",
        "Trend over time. One line per series.",
        required=("x", "y"),
        optional=("color", "facet", "emphasis"),
        color_job="categorical",
        continuous_x=True,
    ),
    "area": _m(
        "area",
        "Trend over time for a single series, or a stacked part-to-whole over time.",
        required=("x", "y"),
        optional=("color", "facet"),
        color_job="categorical",
        continuous_x=True,
    ),
    "scatter": _m(
        "scatter",
        "Relationship between two continuous measures.",
        required=("x", "y"),
        optional=("color", "size", "facet"),
        color_job="categorical",
        pairs="all",
        continuous_x=True,
    ),
    "bubble": _m(
        "bubble",
        "Scatter with a third measure encoded as mark area.",
        required=("x", "y", "size"),
        optional=("color", "facet"),
        color_job="categorical",
        pairs="all",
        continuous_x=True,
    ),
    "heatmap": _m(
        "heatmap",
        "Magnitude over a two-dimensional grid of categories.",
        required=("x", "y", "color"),
        optional=(),
        color_job="sequential",
        pairs="all",
    ),
    "histogram": _m(
        "histogram",
        "Distribution of one continuous measure.",
        required=("x",),
        optional=("color", "facet"),
        color_job="sequential",
        continuous_x=True,
    ),
    "box": _m(
        "box",
        "Distribution of a measure across categories, five-number summary.",
        required=("x", "y"),
        optional=("color", "facet"),
        color_job="categorical",
    ),
    "dumbbell": _m(
        "dumbbell",
        "Before -> after per item. One hue, two shades, connected.",
        required=("x", "y", "color"),
        optional=("facet",),
        color_job="sequential",
    ),
    "stat_tile": _m(
        "stat_tile",
        "A single headline value, optionally with a delta and a sparkline.",
        required=("y",),
        optional=("x", "color"),
        color_job="none",
    ),
    "kpi_row": _m(
        "kpi_row",
        "A row of stat tiles - the honest form for a handful of headline numbers.",
        required=("y",),
        optional=("x", "color"),
        color_job="none",
    ),
    "meter": _m(
        "meter",
        "A single ratio against a limit, on a same-ramp track.",
        required=("y",),
        optional=("x",),
        color_job="sequential",
    ),
    "table": _m(
        "table",
        "The right answer above ~7 meaningful classes, and the accessibility "
        "fallback for every other form.",
        required=(),
        optional=("x", "y", "color", "size", "facet"),
        color_job="none",
    ),
    "pie": _m(
        "pie",
        "Part-to-whole as angle.",
        required=("x", "y"),
        optional=(),
        color_job="categorical",
        pairs="all",
        discouraged="angle is read less accurately than length - a stacked bar "
        "(or a plain bar, if the parts are being compared rather than summed) "
        "carries the same data more precisely",
    ),
    "donut": _m(
        "donut",
        "Pie with the center punched out.",
        required=("x", "y"),
        optional=(),
        color_job="categorical",
        pairs="all",
        discouraged="inherits every drawback of the pie and removes the one "
        "cue that made the center readable - prefer a stacked bar, or a stat "
        "tile if a single share is the point",
    ),
}

ALL_CHANNELS: tuple[str, ...] = ("x", "y", "color", "size", "facet")


# --------------------------------------------------------------------------
# Encodings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Encoding:
    """One channel binding: which column, aggregated how, sorted how."""

    field: str
    aggregate: str = "none"
    time_unit: str | None = None
    sort: str | None = None  # "asc" | "desc" | None (natural order)
    scale: str = "linear"  # "linear" | "log"
    title: str | None = None
    bins: int | None = None  # histogram only

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise SpecError("Encoding.field must be a non-empty column name")
        if self.aggregate not in AGGREGATES:
            raise SpecError(
                f"unknown aggregate {self.aggregate!r}; expected one of {', '.join(AGGREGATES)}"
            )
        if self.time_unit is not None and self.time_unit not in TIME_UNITS:
            raise SpecError(
                f"unknown time_unit {self.time_unit!r}; expected one of {', '.join(TIME_UNITS)}"
            )
        if self.sort not in (None, "asc", "desc"):
            raise SpecError(f"sort must be 'asc', 'desc', or None; got {self.sort!r}")
        if self.scale not in ("linear", "log"):
            raise SpecError(f"scale must be 'linear' or 'log'; got {self.scale!r}")
        if self.bins is not None and self.bins < 1:
            raise SpecError(f"bins must be >= 1; got {self.bins}")

    @property
    def label(self) -> str:
        """Human-facing axis/legend title for this channel."""
        if self.title:
            return self.title
        base = self.field.replace("_", " ").strip()
        base = base[:1].upper() + base[1:] if base else base
        if self.time_unit:
            return f"{base} ({self.time_unit})"
        if self.aggregate not in ("none", "count"):
            return f"{self.aggregate.capitalize()} of {base}"
        if self.aggregate == "count":
            return "Count"
        return base

    def to_dict(self) -> dict[str, Any] | str:
        """Minimal JSON form: the bare column name when nothing else is set."""
        out = {
            name: getattr(self, name)
            for name, f in self.__dataclass_fields__.items()
            if getattr(self, name) != f.default
        }
        if set(out) == {"field"}:
            return self.field
        return out

    @classmethod
    def coerce(cls, value: "Encoding | Mapping[str, Any] | str | None") -> "Encoding | None":
        """Accept an Encoding, a mapping, a bare column name, or None.

        The bare-string form exists because it is what an agent writes first:
        ``{"x": "region"}`` should not need to become
        ``{"x": {"field": "region"}}`` to be legal.
        """
        if value is None:
            return None
        if isinstance(value, Encoding):
            return value
        if isinstance(value, str):
            return cls(field=value)
        if isinstance(value, Mapping):
            unknown = set(value) - {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            if unknown:
                raise SpecError(
                    f"unknown Encoding key(s): {', '.join(sorted(unknown))}"
                    + did_you_mean(
                        sorted(unknown)[0], [f for f in cls.__dataclass_fields__]
                    )
                )
            return cls(**dict(value))
        raise SpecError(f"cannot read an Encoding from {type(value).__name__}")


@dataclass(frozen=True)
class Filter:
    """A row predicate applied before aggregation."""

    field: str
    op: str
    value: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise SpecError("Filter.field must be a non-empty column name")
        if self.op not in FILTER_OPS:
            raise SpecError(
                f"unknown filter op {self.op!r}; expected one of {', '.join(FILTER_OPS)}"
            )
        if self.op in ("in", "not in") and not isinstance(self.value, (list, tuple, set)):
            raise SpecError(f"filter op {self.op!r} needs a sequence value, got {self.value!r}")
        if self.op == "between":
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                raise SpecError("filter op 'between' needs a [low, high] pair")
        if self.op in ("isnull", "notnull") and self.value is not None:
            raise SpecError(f"filter op {self.op!r} takes no value")

    @classmethod
    def coerce(cls, value: "Filter | Mapping[str, Any]") -> "Filter":
        if isinstance(value, Filter):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise SpecError(f"cannot read a Filter from {type(value).__name__}")


# --------------------------------------------------------------------------
# The spec
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChartSpec:
    """A complete, validated description of one chart.

    Frozen on purpose: a spec that mutates between validation and rendering is
    a spec that was never really validated. Use :meth:`with_changes` to derive
    a modified copy.
    """

    chart: str
    x: Encoding | None = None
    y: Encoding | None = None
    color: Encoding | None = None
    size: Encoding | None = None
    facet: Encoding | None = None

    filters: tuple[Filter, ...] = ()
    limit: int | None = None
    """Top-N cap on the primary categorical channel. Always surfaced in the
    caption when applied - a silently truncated chart reads as complete."""

    stack: bool | None = None
    normalize: bool = False
    """Stack to 100% rather than to the raw total."""

    orientation: Literal["auto", "horizontal", "vertical"] = "auto"
    mode: Literal["light", "dark"] = "light"
    palette: tuple[str, ...] | None = None
    """Explicit palette override. Still validated - an override is not an
    exemption."""

    emphasis: str | None = None
    """Value of the color/x field to highlight, with everything else pushed to
    the de-emphasis gray. The most underused form and often the honest answer
    to 'make this chart clearer'."""

    baseline: float | None = None
    """Reference line, and the pivot for diverging forms."""

    title: str | None = None
    subtitle: str | None = None
    caption: str | None = None
    source: str | None = None

    width: int = 720
    height: int = 440
    show_table: bool = True
    """Emit the accessible table view alongside the chart. On by default: it is
    the fallback that makes every other accessibility claim hold."""

    query: str | None = None
    """The natural-language question this spec came from, kept for provenance."""

    notes: tuple[str, ...] = ()
    """Advisories accumulated by the recommender/validator - series folded into
    'Other', a top-N applied, a discouraged form honored on request."""

    def __post_init__(self) -> None:
        if self.chart not in CHART_TYPES:
            raise SpecError(
                f"unknown chart type {self.chart!r}"
                + did_you_mean(self.chart, CHART_TYPES)
                + f"; supported: {', '.join(sorted(CHART_TYPES))}"
            )
        # Coerce channel shorthand in place (frozen dataclass -> object.__setattr__).
        for channel in ALL_CHANNELS:
            object.__setattr__(self, channel, Encoding.coerce(getattr(self, channel)))
        object.__setattr__(
            self, "filters", tuple(Filter.coerce(f) for f in self.filters)
        )
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.palette is not None:
            object.__setattr__(self, "palette", tuple(self.palette))

        if self.limit is not None and self.limit < 1:
            raise SpecError(f"limit must be >= 1; got {self.limit}")
        if self.mode not in ("light", "dark"):
            raise SpecError(f"mode must be 'light' or 'dark'; got {self.mode!r}")
        if self.orientation not in ("auto", "horizontal", "vertical"):
            raise SpecError(f"bad orientation {self.orientation!r}")
        if self.width < 120 or self.height < 80:
            raise SpecError(
                f"width/height must be at least 120x80; got {self.width}x{self.height}"
            )

        meta = self.meta
        missing = [c for c in meta.required if getattr(self, c) is None]
        if missing:
            raise SpecError(
                f"chart {self.chart!r} requires the {', '.join(missing)} "
                f"channel(s): {meta.summary}"
            )
        allowed = set(meta.required) | set(meta.optional) | {"emphasis"}
        extra = [c for c in ALL_CHANNELS if getattr(self, c) is not None and c not in allowed]
        if extra:
            raise SpecError(
                f"chart {self.chart!r} does not use the {', '.join(extra)} channel(s). "
                f"It was not dropped silently because that would render a chart you "
                f"did not ask for; remove it or pick a form that uses it."
            )
        if self.normalize and not self.stack:
            raise SpecError("normalize=True only means something with stack=True")

    # -- derived ---------------------------------------------------------

    @property
    def meta(self) -> ChartMeta:
        """Registry entry for this chart form."""
        return CHART_TYPES[self.chart]

    @property
    def color_job(self) -> ColorJob:
        """Which job color does here - ``emphasis`` overrides the form default."""
        if self.emphasis is not None:
            return "emphasis"
        return self.meta.color_job

    @property
    def pairs(self) -> Literal["adjacent", "all"]:
        return self.meta.pairs

    @property
    def is_horizontal(self) -> bool:
        """Resolve ``orientation='auto'`` for bar-family forms."""
        if self.orientation != "auto":
            return self.orientation == "horizontal"
        return self.chart in ("bar", "diverging_bar", "dumbbell")

    def with_changes(self, **kwargs: Any) -> "ChartSpec":
        """Return a validated copy with *kwargs* replaced."""
        return replace(self, **kwargs)

    def with_note(self, note: str) -> "ChartSpec":
        """Return a copy with *note* appended (deduplicated)."""
        if note in self.notes:
            return self
        return replace(self, notes=self.notes + (note,))

    def fields_used(self) -> list[str]:
        """Every column this spec touches, deduplicated, in channel order."""
        out: list[str] = []
        for channel in ALL_CHANNELS:
            enc = getattr(self, channel)
            if enc is not None and enc.field not in out:
                out.append(enc.field)
        for f in self.filters:
            if f.field not in out:
                out.append(f.field)
        return out

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON dict.

        Fields still at their dataclass default are omitted, so the output is
        the minimal spec that round-trips back to this object rather than a
        wall of nulls. ``chart`` is always emitted and always first.
        """
        out: dict[str, Any] = {"chart": self.chart}
        for name, f in self.__dataclass_fields__.items():
            if name == "chart":
                continue
            value = getattr(self, name)
            if value is None or value == () or value == [] or value == f.default:
                continue
            if isinstance(value, Encoding):
                value = value.to_dict()
            elif name == "filters":
                value = [
                    {k: v for k, v in asdict(flt).items() if k != "value" or v is not None}
                    for flt in value
                ]
            elif isinstance(value, tuple):
                value = list(value)
            out[name] = value
        return out

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChartSpec":
        """Build a spec from a mapping, rejecting unknown keys.

        Unknown keys are an error rather than a shrug: an agent that
        misremembers a key name should find out immediately, not ship a chart
        that quietly ignored half its instructions.
        """
        if not isinstance(data, Mapping):
            raise SpecError(f"ChartSpec.from_dict needs a mapping, got {type(data).__name__}")
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            first = sorted(unknown)[0]
            raise SpecError(
                f"unknown ChartSpec key(s): {', '.join(sorted(unknown))}"
                + did_you_mean(first, known)
            )
        payload = dict(data)
        if "filters" in payload and payload["filters"] is not None:
            payload["filters"] = tuple(Filter.coerce(f) for f in payload["filters"])
        if "notes" in payload and payload["notes"] is not None:
            payload["notes"] = tuple(payload["notes"])
        return cls(**payload)

    @classmethod
    def from_json(cls, text: str) -> "ChartSpec":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SpecError(f"chart spec is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    def describe(self) -> str:
        """One-line human summary, used in captions and CLI output."""
        bits = [self.chart]
        for channel in ALL_CHANNELS:
            enc = getattr(self, channel)
            if enc is not None:
                bits.append(f"{channel}={enc.label}")
        if self.filters:
            bits.append(f"{len(self.filters)} filter(s)")
        if self.limit:
            bits.append(f"top {self.limit}")
        return " | ".join(bits)


def chart_catalog() -> list[dict[str, Any]]:
    """Machine-readable catalog of every supported form.

    Exposed through the CLI (``dataviz-agent charts``) so an agent can ask what
    is available rather than guess a chart name and get an error.
    """
    return [
        {
            "chart": name,
            "summary": meta.summary,
            "required": list(meta.required),
            "optional": list(meta.optional),
            "color_job": meta.color_job,
            "pairs": meta.pairs,
            "discouraged": meta.discouraged,
        }
        for name, meta in sorted(CHART_TYPES.items())
    ]
