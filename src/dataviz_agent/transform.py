"""Executing a spec against a DataFrame.

By the time a :class:`~dataviz_agent.spec.ChartSpec` arrives here, intent is
data: this module never decides what the chart should be, it only carries out
what the spec says - filter, bin time, aggregate, sort, cap.

Two rules govern everything below:

**Nothing is dropped silently.** A top-N, a folded "Other" bucket, rows lost to
a filter or to null keys - each one appends a note that the renderer puts in
the caption. A truncated chart that looks complete is the most common way a
correct pipeline still misleads a reader.

**The result is chart-ready.** One row per mark, columns named by channel, in
draw order. The renderer should never have to re-aggregate or re-sort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .errors import SpecError
from .profile import DataProfile, profile_frame
from .spec import ALL_CHANNELS, ChartSpec, Encoding, Filter

__all__ = ["TransformResult", "apply_spec", "OTHER_LABEL"]

OTHER_LABEL = "Other"

#: Series beyond this many are folded into "Other" by default. It is the
#: palette's categorical cap, not an arbitrary round number: past it, two
#: series that land adjacent can no longer be told apart under a colorblind
#: simulation, and a legend stops being scannable.
DEFAULT_MAX_SERIES = 6

_PANDAS_AGG = {
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "min": "min",
    "max": "max",
    "count": "size",
    "nunique": "nunique",
    "std": "std",
}

_TIME_PERIOD = {
    "year": "Y",
    "quarter": "Q",
    "month": "M",
    "week": "W",
    "day": "D",
    "hour": "h",
}


@dataclass(frozen=True)
class TransformResult:
    """The chart-ready frame plus everything the caption has to admit to."""

    frame: Any
    """One row per mark. Columns are named for the channels they feed."""

    spec: ChartSpec
    """The spec as executed, carrying any notes this pass added."""

    columns: dict[str, str] = field(default_factory=dict)
    """Channel -> column name in :attr:`frame`."""

    n_rows_in: int = 0
    n_rows_used: int = 0

    @property
    def notes(self) -> tuple[str, ...]:
        return self.spec.notes

    def column_for(self, channel: str) -> str | None:
        return self.columns.get(channel)

    def series_names(self) -> list[str]:
        """Distinct color-channel values, in draw order."""
        col = self.columns.get("color")
        if col is None:
            return []
        return list(dict.fromkeys(self.frame[col].tolist()))

    def to_records(self) -> list[dict[str, Any]]:
        """Plain rows - the accessible table view, and the CLI's JSON output."""
        return self.frame.to_dict(orient="records")


def apply_spec(
    df: Any,
    spec: ChartSpec,
    *,
    profile: DataProfile | None = None,
    max_series: int = DEFAULT_MAX_SERIES,
) -> TransformResult:
    """Run *spec* against *df* and return a chart-ready frame.

    Parameters
    ----------
    profile:
        Reused if supplied; profiling is not free on a wide frame and the
        caller usually has one already.
    max_series:
        Cap on distinct color values before the tail is folded into "Other".
        Pass ``0`` to keep every series - legitimate for a table, and a
        mistake for anything with a legend.
    """
    import pandas as pd

    profile = profile or profile_frame(df)
    _check_columns(spec, profile)

    n_rows_in = len(df)
    work = df.copy()
    spec_out = spec

    for flt in spec.filters:
        work, spec_out = _apply_filter(work, flt, spec_out, profile)

    if work.empty:
        raise SpecError(
            "no rows left after filtering; "
            + (
                f"check the filter values against the data "
                f"(e.g. {profile.column(spec.filters[0].field).sample[:3]})"
                if spec.filters
                else "the input was empty"
            )
        )

    # Bin time before grouping, so "revenue by month" groups on the month and
    # not on 4,000 distinct timestamps.
    for channel in ALL_CHANNELS:
        enc: Encoding | None = getattr(spec, channel)
        if enc is not None and enc.time_unit:
            work = work.assign(
                **{enc.field: _floor_time(work[enc.field], enc.time_unit, pd)}
            )

    work, spec_out = _drop_null_keys(work, spec, spec_out)
    frame, columns = _aggregate(work, spec, pd)
    n_rows_used = len(work)

    frame, spec_out = _fold_series(frame, spec, spec_out, columns, max_series, pd)
    frame, spec_out = _sort_and_cap(frame, spec, spec_out, columns)

    return TransformResult(
        frame=frame.reset_index(drop=True),
        spec=spec_out,
        columns=columns,
        n_rows_in=n_rows_in,
        n_rows_used=n_rows_used,
    )


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


def _check_columns(spec: ChartSpec, profile: DataProfile) -> None:
    """Fail on every unknown column at once, before doing any work."""
    unknown = [name for name in spec.fields_used() if name not in profile]
    if not unknown:
        return
    # profile.column() raises with a suggestion and the full schema.
    profile.column(unknown[0])


def _apply_filter(
    work: Any, flt: Filter, spec: ChartSpec, profile: DataProfile
) -> tuple[Any, ChartSpec]:
    series = work[flt.field]
    op = flt.op
    if op == "==":
        mask = series == flt.value
    elif op == "!=":
        mask = series != flt.value
    elif op == ">":
        mask = series > flt.value
    elif op == ">=":
        mask = series >= flt.value
    elif op == "<":
        mask = series < flt.value
    elif op == "<=":
        mask = series <= flt.value
    elif op == "in":
        mask = series.isin(list(flt.value))
    elif op == "not in":
        mask = ~series.isin(list(flt.value))
    elif op == "contains":
        mask = series.astype(str).str.contains(str(flt.value), case=False, na=False)
    elif op == "between":
        low, high = flt.value
        mask = series.between(low, high)
    elif op == "isnull":
        mask = series.isna()
    elif op == "notnull":
        mask = series.notna()
    else:  # pragma: no cover - Filter.__post_init__ rejects anything else
        raise SpecError(f"unhandled filter op {op!r}")

    kept = int(mask.sum())
    if kept == 0:
        col = profile.column(flt.field)
        raise SpecError(
            f"filter {flt.field} {op} {flt.value!r} matched no rows; "
            f"{flt.field} looks like {list(col.sample[:5])}"
        )
    dropped = len(work) - kept
    if dropped:
        spec = spec.with_note(
            f"filtered to {flt.field} {op} {_fmt_value(flt.value)} "
            f"({dropped:,} of {len(work):,} rows excluded)"
        )
    return work[mask], spec


def _drop_null_keys(work: Any, spec: ChartSpec, spec_out: ChartSpec) -> tuple[Any, ChartSpec]:
    """Remove rows with no value on a grouping channel.

    A null category becomes a nameless band that readers charitably read as
    zero. Better to drop it and say so.
    """
    keys = [
        enc.field
        for channel, enc in _channels(spec)
        if channel in ("x", "color", "facet") and enc is not None
    ]
    if not keys:
        return work, spec_out
    mask = work[keys].notna().all(axis=1)
    dropped = int((~mask).sum())
    if dropped:
        spec_out = spec_out.with_note(
            f"{dropped:,} row(s) dropped for a missing {', '.join(keys)} value"
        )
    return work[mask], spec_out


def _aggregate(work: Any, spec: ChartSpec, pd: Any) -> tuple[Any, dict[str, str]]:
    """Group and aggregate into one row per mark."""
    columns: dict[str, str] = {}
    group_channels = [
        (channel, enc)
        for channel, enc in _channels(spec)
        if channel in ("x", "color", "facet")
        and enc is not None
        and enc.aggregate == "none"
    ]
    measures = [
        (channel, enc)
        for channel, enc in _channels(spec)
        if enc is not None and enc.aggregate != "none"
    ]

    for channel, enc in _channels(spec):
        if enc is not None:
            columns[channel] = enc.field

    # "count the orders in each region" binds the same column to x and y. The
    # aggregate needs its own name or it collides with the group key.
    group_fields = {enc.field for _, enc in group_channels}
    out_names: dict[str, str] = {}
    for channel, enc in measures:
        name = enc.field
        if name in group_fields or name in out_names.values():
            name = f"{enc.field}_{enc.aggregate}"
        out_names[channel] = name
        columns[channel] = name

    if not measures:
        # Nothing to aggregate: the spec asks for the rows themselves
        # (scatter, an already-summarized table).
        keep = [enc.field for _, enc in _channels(spec) if enc is not None]
        return work[list(dict.fromkeys(keep))].copy(), columns

    if not group_channels:
        # A single aggregate with nothing to group by - one mark.
        row = {
            out_names[channel]: _agg_series(work[enc.field], enc.aggregate)
            for channel, enc in measures
        }
        return pd.DataFrame([row]), columns

    keys = [enc.field for _, enc in group_channels]
    grouped = work.groupby(keys, dropna=False, observed=True)
    pieces = {}
    for channel, enc in measures:
        how = _PANDAS_AGG[enc.aggregate]
        pieces[out_names[channel]] = (
            grouped.size() if how == "size" else grouped[enc.field].agg(how)
        )
    frame = pd.DataFrame(pieces).reset_index()
    return frame, columns


def _agg_series(series: Any, how: str) -> Any:
    if how == "count":
        return int(series.notna().sum())
    if how == "nunique":
        return int(series.nunique())
    return getattr(series, _PANDAS_AGG[how])()


def _fold_series(
    frame: Any,
    spec: ChartSpec,
    spec_out: ChartSpec,
    columns: dict[str, str],
    max_series: int,
    pd: Any,
) -> tuple[Any, ChartSpec]:
    """Fold color values past the cap into a single "Other" series."""
    color_col = columns.get("color")
    if not max_series or color_col is None or spec.chart == "table":
        return frame, spec_out
    value_col = columns.get("y") or columns.get("size")
    series_names = list(dict.fromkeys(frame[color_col].tolist()))
    if len(series_names) <= max_series:
        return frame, spec_out

    if value_col is not None and pd.api.types.is_numeric_dtype(frame[value_col]):
        ranked = (
            frame.groupby(color_col, observed=True)[value_col]
            .sum()
            .sort_values(ascending=False)
        )
        keep = list(ranked.index[: max_series - 1])
    else:
        keep = series_names[: max_series - 1]

    folded = frame.assign(
        **{
            color_col: frame[color_col].where(
                frame[color_col].isin(keep), OTHER_LABEL
            )
        }
    )
    group_cols = [
        col
        for channel, col in columns.items()
        if channel in ("x", "color", "facet")
    ]
    if value_col is not None and group_cols:
        agg = {
            col: "sum" if pd.api.types.is_numeric_dtype(folded[col]) else "first"
            for col in folded.columns
            if col not in group_cols
        }
        folded = folded.groupby(group_cols, dropna=False, observed=True).agg(agg).reset_index()

    n_folded = len(series_names) - len(keep)
    return folded, spec_out.with_note(
        f"{n_folded} smaller series folded into '{OTHER_LABEL}' "
        f"(a legend past {max_series} colors stops being readable)"
    )


def _sort_and_cap(
    frame: Any, spec: ChartSpec, spec_out: ChartSpec, columns: dict[str, str]
) -> tuple[Any, ChartSpec]:
    """Order the marks, then apply any top-N - in that order, so the cap keeps
    the rows the sort says matter."""
    sort_channel, ascending = _sort_plan(spec, columns)
    if sort_channel is not None:
        frame = frame.sort_values(columns[sort_channel], ascending=ascending, kind="mergesort")

    if spec.limit is None:
        return frame, spec_out

    cap_col = columns.get("x") if not spec.meta.continuous_x else None
    cap_col = cap_col or columns.get("color") or columns.get("x")
    if cap_col is None:
        return frame, spec_out

    keys = list(dict.fromkeys(frame[cap_col].tolist()))
    if len(keys) <= spec.limit:
        return frame, spec_out
    kept = keys[: spec.limit]
    frame = frame[frame[cap_col].isin(kept)]
    return frame, spec_out.with_note(
        f"showing the top {spec.limit} of {len(keys):,} {cap_col} values"
    )


def _sort_plan(spec: ChartSpec, columns: dict[str, str]) -> tuple[str | None, bool]:
    """Which channel orders the marks, and which way.

    An explicit ``sort`` on any encoding wins. Otherwise a categorical chart
    sorts by its measure descending - ranked order is what makes a bar chart
    readable - while a time axis keeps chronological order, because sorting a
    trend by value destroys the trend.
    """
    for channel in ("y", "x", "color", "size"):
        enc: Encoding | None = getattr(spec, channel, None)
        if enc is not None and enc.sort:
            return channel, enc.sort == "asc"
    if spec.meta.continuous_x:
        return ("x", True) if "x" in columns else (None, True)
    if "y" in columns and spec.y is not None and spec.y.aggregate != "none":
        return "y", False
    return ("x", True) if "x" in columns else (None, True)


def _floor_time(series: Any, unit: str, pd: Any) -> Any:
    values = pd.to_datetime(series, errors="coerce")
    return values.dt.to_period(_TIME_PERIOD[unit]).dt.to_timestamp()


def _channels(spec: ChartSpec):
    for channel in ALL_CHANNELS:
        yield channel, getattr(spec, channel)


def _fmt_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        shown = ", ".join(str(v) for v in list(value)[:3])
        return f"[{shown}{', ...' if len(value) > 3 else ''}]"
    return str(value)
