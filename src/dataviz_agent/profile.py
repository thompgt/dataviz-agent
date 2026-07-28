"""Profiling - what the data actually is, before anyone decides what to draw.

Every later stage reads from a :class:`DataProfile` rather than from the
DataFrame directly. That is the point: the recommender should not be poking at
dtypes, and the query parser should not be guessing whether ``region`` is a
category or a free-text note. Both ask the profile, and the profile decided
once, from evidence.

The judgement calls live here, and they are deliberately conservative:

* A numeric column that is really a set of labels (``store_id``, a 0/1 flag)
  is classified by *behaviour*, not by dtype - charting the mean of a store id
  is a bug that a dtype check will never catch.
* A string column that never repeats is an identifier, not a category. Nothing
  useful comes of a bar chart with one bar per row.
* Anything that looks like a date is parsed as one, but only if nearly every
  value parses. A column where a third of the values happen to look like dates
  is a text column with some dates in it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

from .errors import IngestError, SpecError, did_you_mean

__all__ = [
    "ColumnRole",
    "ColumnProfile",
    "DataProfile",
    "profile_frame",
]

ColumnRole = Literal[
    "temporal",  # dates / datetimes - the natural x for a trend
    "quantitative",  # a measure worth aggregating
    "categorical",  # a small set of repeating labels
    "boolean",  # two states
    "identifier",  # (near-)unique per row; never a category, never a measure
    "text",  # free text - high cardinality, long values
    "empty",  # entirely null; carried so errors can say so out loud
]

#: A column with more distinct values than this is not a chart axis worth
#: drawing directly - past here the recommender reaches for a top-N or a table.
MAX_CATEGORY_CARDINALITY = 50

#: Above this share of distinct values, a string column is an identifier
#: rather than a category.
IDENTIFIER_UNIQUE_RATIO = 0.9

#: A numeric column with at most this many distinct values, all integral, is
#: read as a set of codes rather than as a measure.
NUMERIC_CODE_MAX_UNIQUE = 12

#: Share of non-null values that must parse as dates before a text column is
#: treated as temporal.
DATE_PARSE_THRESHOLD = 0.95


@dataclass(frozen=True)
class ColumnProfile:
    """What one column is, and what it can honestly be used for."""

    name: str
    role: ColumnRole
    dtype: str
    n_rows: int
    n_missing: int
    n_unique: int
    sample: tuple[Any, ...] = ()
    """A few real values - what an agent needs to map "revenue" onto a column,
    and what makes a filter value like ``"West"`` checkable before it runs."""

    min: Any = None
    max: Any = None
    mean: float | None = None
    has_negative: bool = False
    """Drives the zero-baseline and diverging-form decisions."""

    is_monotonic: bool = False
    """True for a time column already in order - a trend can be drawn as-is."""

    top_values: tuple[tuple[Any, int], ...] = ()
    """The most common values with their counts, for categorical columns."""

    notes: tuple[str, ...] = ()
    """Why this column was classified the way it was, when the answer was not
    obvious from the dtype. Surfaced in ``describe()`` so a puzzling
    recommendation can be traced back to the evidence behind it."""

    # -- capability questions the rest of the package asks ----------------

    @property
    def missing_ratio(self) -> float:
        return self.n_missing / self.n_rows if self.n_rows else 0.0

    @property
    def is_measure(self) -> bool:
        """Can this be aggregated into a y value?"""
        return self.role == "quantitative"

    @property
    def is_dimension(self) -> bool:
        """Can this be grouped by?"""
        return self.role in ("categorical", "boolean", "temporal")

    @property
    def is_drawable_axis(self) -> bool:
        """Small enough to become a set of bands without a top-N first."""
        return self.is_dimension and (
            self.role == "temporal" or self.n_unique <= MAX_CATEGORY_CARDINALITY
        )

    def describe(self) -> str:
        bits = [f"{self.name}: {self.role}", f"{self.n_unique} distinct"]
        if self.n_missing:
            bits.append(f"{self.missing_ratio:.0%} missing")
        if self.role == "quantitative" and self.min is not None:
            bits.append(f"range {_fmt(self.min)}..{_fmt(self.max)}")
        elif self.role == "temporal" and self.min is not None:
            bits.append(f"{self.min} .. {self.max}")
        elif self.top_values:
            shown = ", ".join(str(v) for v, _ in self.top_values[:3])
            bits.append(f"e.g. {shown}")
        if self.notes:
            bits.append("; ".join(self.notes))
        return " | ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "dtype": self.dtype,
            "n_unique": self.n_unique,
            "n_missing": self.n_missing,
            "sample": [_jsonable(v) for v in self.sample],
            "min": _jsonable(self.min),
            "max": _jsonable(self.max),
            "top_values": [[_jsonable(v), n] for v, n in self.top_values],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class DataProfile:
    """The profiled shape of one table."""

    n_rows: int
    columns: tuple[ColumnProfile, ...]
    truncated: bool = False
    """True when profiling sampled the frame rather than reading all of it."""

    _by_name: dict[str, ColumnProfile] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "_by_name", {c.name: c for c in self.columns})

    # -- lookup ----------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def get(self, name: str) -> ColumnProfile | None:
        return self._by_name.get(name)

    def column(self, name: str) -> ColumnProfile:
        """Look up a column, failing with the schema and a suggestion.

        The caller here is usually a model retrying a spec, so the error has to
        carry enough to fix it in one step rather than send it back to re-read
        the schema.
        """
        found = self._by_name.get(name)
        if found is not None:
            return found
        raise SpecError(
            f"no column named {name!r}"
            + did_you_mean(name, self.names)
            + f"; available: {', '.join(self.names)}"
        )

    def resolve(self, name: str) -> ColumnProfile:
        """Like :meth:`column`, but tolerant of case and separator differences.

        ``Net Revenue``, ``net_revenue`` and ``netrevenue`` all find the same
        column - the kinds of near-miss a question phrased in English produces.
        Exact matches always win, so a table with both ``Region`` and ``region``
        still behaves predictably.
        """
        found = self._by_name.get(name)
        if found is not None:
            return found
        key = _normalize(name)
        matches = [c for c in self.columns if _normalize(c.name) == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise SpecError(
                f"{name!r} matches more than one column ({', '.join(c.name for c in matches)}); "
                "use the exact column name"
            )
        return self.column(name)  # raises with the full schema

    # -- role queries the recommender and parser lean on ------------------

    def by_role(self, *roles: str) -> list[ColumnProfile]:
        return [c for c in self.columns if c.role in roles]

    @property
    def measures(self) -> list[ColumnProfile]:
        return self.by_role("quantitative")

    @property
    def dimensions(self) -> list[ColumnProfile]:
        """Groupable columns, lowest cardinality first - the order a chart
        wants them in, since the smallest set makes the best color channel."""
        dims = self.by_role("categorical", "boolean", "temporal")
        return sorted(dims, key=lambda c: (c.role != "temporal", c.n_unique))

    @property
    def temporal(self) -> list[ColumnProfile]:
        return self.by_role("temporal")

    def describe(self) -> str:
        head = f"{self.n_rows:,} rows x {len(self.columns)} columns"
        if self.truncated:
            head += " (profiled from a sample)"
        return "\n".join([head, *(f"  {c.describe()}" for c in self.columns)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "truncated": self.truncated,
            "columns": [c.to_dict() for c in self.columns],
        }


# --------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------


def profile_frame(df: Any, *, sample: int | None = 50_000) -> DataProfile:
    """Profile a pandas DataFrame.

    Parameters
    ----------
    df:
        The frame to profile.
    sample:
        Row cap for the statistics. Large frames are profiled from the head
        plus a random sample rather than in full; the resulting profile is
        flagged ``truncated`` so nothing downstream reports sampled counts as
        exact. Pass ``None`` to profile every row.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise IngestError(
            f"profile_frame needs a pandas DataFrame, got {type(df).__name__}"
        )
    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        raise IngestError(
            f"duplicate column name(s): {', '.join(map(str, dupes))}. "
            "Rename them before profiling - a spec naming one of these could "
            "not say which was meant."
        )

    n_rows = int(len(df))
    truncated = sample is not None and n_rows > sample
    frame = df.sample(n=sample, random_state=0) if truncated else df

    columns = tuple(
        _profile_column(str(name), frame[name], n_rows=n_rows) for name in df.columns
    )
    return DataProfile(n_rows=n_rows, columns=columns, truncated=truncated)


def _profile_column(name: str, series: Any, *, n_rows: int) -> ColumnProfile:
    import pandas as pd

    dtype = str(series.dtype)
    non_null = series.dropna()
    n_missing = int(len(series) - len(non_null))
    notes: list[str] = []

    if non_null.empty:
        return ColumnProfile(
            name=name,
            role="empty",
            dtype=dtype,
            n_rows=n_rows,
            n_missing=n_rows,
            n_unique=0,
            notes=("every value is null",),
        )

    n_unique = int(non_null.nunique())
    sample = tuple(non_null.drop_duplicates().head(5).tolist())

    role, coerced, role_notes = _classify(name, non_null, n_unique, dtype)
    notes.extend(role_notes)
    values = coerced if coerced is not None else non_null

    kwargs: dict[str, Any] = {}
    if role == "quantitative":
        kwargs.update(
            min=_scalar(values.min()),
            max=_scalar(values.max()),
            mean=float(values.mean()),
            has_negative=bool((values < 0).any()),
        )
    elif role == "temporal":
        kwargs.update(
            min=str(values.min()),
            max=str(values.max()),
            is_monotonic=bool(values.is_monotonic_increasing),
        )
        sample = tuple(str(v) for v in values.drop_duplicates().head(5))
    if role in ("categorical", "boolean"):
        counts = values.value_counts().head(10)
        kwargs["top_values"] = tuple(
            (_scalar(v), int(n)) for v, n in counts.items()
        )
        if n_unique > MAX_CATEGORY_CARDINALITY:
            notes.append(
                f"{n_unique} distinct values - needs a top-N or a table, "
                "not one band per value"
            )

    return ColumnProfile(
        name=name,
        role=role,
        dtype=dtype,
        n_rows=n_rows,
        n_missing=n_missing,
        n_unique=n_unique,
        sample=sample,
        notes=tuple(notes),
        **kwargs,
    )


def _classify(
    name: str, values: Any, n_unique: int, dtype: str
) -> tuple[ColumnRole, Any, list[str]]:
    """Decide a column's role.

    Returns the role, a coerced series when parsing changed the values (dates
    read out of strings), and any notes explaining a non-obvious call.
    """
    import pandas as pd
    from pandas.api import types as pdt

    notes: list[str] = []

    if pdt.is_bool_dtype(values):
        return "boolean", None, notes

    if pdt.is_datetime64_any_dtype(values) or isinstance(values.dtype, pd.PeriodDtype):
        return "temporal", values, notes

    if pdt.is_numeric_dtype(values):
        if n_unique <= 2 and set(map(float, pd.unique(values))) <= {0.0, 1.0}:
            return "boolean", None, ["0/1 column read as a flag, not a measure"]
        integral = _is_integral(values)
        if integral and n_unique <= NUMERIC_CODE_MAX_UNIQUE and _looks_like_code(name):
            return (
                "categorical",
                None,
                [
                    f"numeric but only {n_unique} whole values, and named like a "
                    "code - grouped by rather than averaged"
                ],
            )
        if integral and _looks_like_identifier(name) and n_unique == len(values):
            return "identifier", None, ["unique whole numbers named like an id"]
        return "quantitative", None, notes

    # Object/string-like from here down.
    as_str = values.astype(str)
    parsed = _try_datetime(values)
    if parsed is not None:
        return "temporal", parsed, ["parsed from text into dates"]

    numeric = pd.to_numeric(as_str.str.replace(r"[,$%\s]", "", regex=True), errors="coerce")
    if numeric.notna().mean() >= DATE_PARSE_THRESHOLD and n_unique > NUMERIC_CODE_MAX_UNIQUE:
        return (
            "quantitative",
            numeric.dropna(),
            ["numbers stored as text (currency/thousands formatting stripped)"],
        )

    unique_ratio = n_unique / len(values)
    mean_len = float(as_str.str.len().mean())
    # A column of distinct strings is an identifier - but on a small frame
    # "distinct" is weak evidence (six rows, six regions), so below the code
    # threshold the name has to back it up.
    if unique_ratio >= IDENTIFIER_UNIQUE_RATIO and (
        n_unique > NUMERIC_CODE_MAX_UNIQUE or _looks_like_identifier(name)
    ):
        role: ColumnRole = "text" if mean_len > 40 else "identifier"
        notes.append(
            f"{unique_ratio:.0%} of values are distinct - "
            + ("free text" if role == "text" else "an identifier, not a category")
        )
        return role, None, notes
    if mean_len > 60:
        return "text", None, ["values average over 60 characters - free text"]
    return "categorical", None, notes


def _try_datetime(values: Any) -> Any | None:
    """Parse to datetimes, or return None if too few values parse."""
    import warnings

    import pandas as pd

    head = values.head(200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            probe = pd.to_datetime(head, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            return None
        if probe.notna().mean() < DATE_PARSE_THRESHOLD:
            return None
        try:
            full = pd.to_datetime(values, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            return None
    if full.notna().mean() < DATE_PARSE_THRESHOLD:
        return None
    return full.dropna()


def _is_integral(values: Any) -> bool:
    from pandas.api import types as pdt

    if pdt.is_integer_dtype(values):
        return True
    try:
        return bool((values.dropna() % 1 == 0).all())
    except TypeError:
        return False


_CODE_HINTS = ("id", "code", "no", "num", "zip", "postal", "year", "month", "week", "day")
_ID_HINTS = ("id", "uuid", "guid", "key", "hash", "ref", "sku", "slug")


def _looks_like_code(name: str) -> bool:
    return any(word in _CODE_HINTS for word in _words(name))


def _looks_like_identifier(name: str) -> bool:
    return any(word in _ID_HINTS for word in _words(name))


def _words(name: str) -> list[str]:
    """Split a column name into lowercase words across any separator style,
    including camelCase: ``orderID`` -> ``['order', 'id']``."""
    spaced = "".join(
        (" " + ch if ch.isupper() else ch) if ch.isalnum() else " " for ch in str(name)
    )
    return spaced.lower().split()


def _normalize(name: str) -> str:
    """Fold case and separators so 'Net Revenue', 'net_revenue' and
    'netrevenue' all compare equal."""
    return "".join(_words(name))


def _scalar(value: Any) -> Any:
    """Unwrap numpy scalars so the profile stays plain-Python."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _jsonable(value: Any) -> Any:
    value = _scalar(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4g}"
    return f"{value:,}" if isinstance(value, int) else str(value)
