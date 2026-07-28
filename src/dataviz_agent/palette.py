"""The palette layer: a validated reference palette plus a runnable validator.

The single most important habit this package enforces is that **the color part
is computable, so it gets computed**. Nothing in the render path picks a color
by taste; every categorical palette that reaches a chart has been through
:func:`validate_palette` first.

The six checks
--------------
1. **Lightness band** - every slot sits in the mode's OKLCh ``L`` band, so no
   slot disappears into the surface or blows out against it.
2. **Chroma floor** - every slot carries enough chroma to read as a hue rather
   than as a gray.
3. **CVD separation** - each pair on the active pairlist stays >= 8 dE apart
   under the worst of protanopia / deuteranopia / tritanopia. 6-8 is a WARN
   band that is legal only with secondary encoding (direct labels, texture,
   shape); below 6 is a FAIL.
4. **Normal-vision floor** - each pair stays >= 15 dE apart unsimulated. This
   one is a hard FAIL and secondary encoding does **not** excuse it: if
   full-color readers cannot separate the pair, the palette is wrong.
5. **Surface contrast** - each slot is checked against the chart surface at
   3:1. Falling short is a WARN that *obligates* the relief rule (visible
   direct labels or a table view); it is not dismissable.
6. **Distinctness** - no two slots are the same color.

Pairlists
---------
Which pairs get checked depends on the chart form:

* ``"adjacent"`` - stacks, bars, lines. Only slots that touch on screen need
  to separate, so only consecutive pairs are gated.
* ``"all"`` - scatter, bubble, choropleth, small multiples. Any two series can
  land next to each other, so all pairs are gated. The reference palette caps
  at **three** slots under this pairlist.

Run it from the CLI::

    dataviz-agent validate-palette "#2a78d6,#eb6834,#1baf7a" --mode light
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Literal, Sequence

from .color import (
    CVD_KINDS,
    GATED_CVD_KINDS,
    CVDKind,
    contrast_ratio,
    delta_e,
    hex_to_oklch,
    is_hex_color,
    parse_hex,
    strip_ws,
    to_hex,
)

__all__ = [
    "Mode",
    "PairList",
    "Severity",
    "CheckResult",
    "PaletteReport",
    "CATEGORICAL",
    "SEQUENTIAL_BLUE",
    "SEQUENTIAL_ORANGE",
    "DIVERGING",
    "STATUS",
    "CHROME",
    "SURFACES",
    "ALL_PAIRS_SERIES_CAP",
    "ADJACENT_SERIES_CAP",
    "categorical",
    "sequential_ramp",
    "diverging_ramp",
    "emphasis_pair",
    "status_color",
    "chrome",
    "validate_palette",
]

Mode = Literal["light", "dark"]
PairList = Literal["adjacent", "all"]
Severity = Literal["PASS", "WARN", "FAIL"]

# --------------------------------------------------------------------------
# Gate thresholds (dE units = OKLab distance x100)
# --------------------------------------------------------------------------

CVD_TARGET = 8.0
"""At or above this, a pair is comfortably separable under dichromacy.

Calibrated against the Machado severity-1.0 simulation in :mod:`.color`, on
``min(protan, deutan)``. Changing the simulation model invalidates this number.
"""

CVD_FLOOR = 6.0
"""Between CVD_FLOOR and CVD_TARGET a pair WARNs: legal only with secondary encoding."""

NORMAL_VISION_FLOOR = 15.0
"""Below this, full-color readers cannot separate the pair. Hard FAIL."""

CONTRAST_TARGET = 3.0
"""Non-text contrast against the chart surface (WCAG 1.4.11)."""

CHROMA_FLOOR = 0.10
"""Below this a slot reads as a neutral rather than as an identity hue."""

LIGHTNESS_BAND: dict[str, tuple[float, float]] = {
    "light": (0.43, 0.77),
    "dark": (0.48, 0.67),
}
"""OKLCh L band per mode. Slots outside it collapse toward the surface.

The dark band is tighter and sits higher: a dark surface leaves less room
below, and the dark column is the same hues *re-stepped* for that surface
rather than an automatic flip of the light column.
"""

ORDINAL_MIN_DELTA_L = 0.06
"""Minimum OKLCh L step between adjacent rungs of an ordinal ramp."""

ORDINAL_SURFACE_FLOOR = 2.0
"""The rung nearest the surface must still clear this WCAG ratio."""

ADJACENT_SERIES_CAP = 8
"""Token ceiling for adjacent forms (stacked bars, grouped bars, multi-line)."""

ALL_PAIRS_SERIES_CAP = 3
"""Cap for all-pairs forms (scatter, bubble, choropleth, small multiples)."""

# --------------------------------------------------------------------------
# The reference palette instance
# --------------------------------------------------------------------------

CATEGORICAL: dict[Mode, tuple[str, ...]] = {
    "light": (
        "#2a78d6",  # 1 blue
        "#eb6834",  # 2 orange
        "#1baf7a",  # 3 aqua
        "#eda100",  # 4 yellow
        "#e87ba4",  # 5 magenta
        "#008300",  # 6 green
        "#4a3aa7",  # 7 violet
        "#e34948",  # 8 red
    ),
    "dark": (
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
}
"""Fixed categorical hue order. **Assigned in order, never cycled.**

A 9th series is never a generated hue - it folds into "Other", facets into
small multiples, or takes a composite encoding (hue x shape).
"""

CATEGORICAL_NAMES = (
    "blue",
    "orange",
    "aqua",
    "yellow",
    "magenta",
    "green",
    "violet",
    "red",
)

SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb",  # 100
    "#b7d3f6",  # 150
    "#9ec5f4",  # 200
    "#86b6ef",  # 250
    "#6da7ec",  # 300
    "#5598e7",  # 350
    "#3987e5",  # 400
    "#2a78d6",  # 450
    "#256abf",  # 500
    "#1c5cab",  # 550
    "#184f95",  # 600
    "#104281",  # 650
    "#0d366b",  # 700
)
"""Default sequential hue, light -> dark, steps 100..700."""

SEQUENTIAL_ORANGE: tuple[str, ...] = (
    "#fde3d4",
    "#fbd0b8",
    "#f9bd9b",
    "#f7aa7f",
    "#f49763",
    "#f18448",
    "#eb6834",
    "#d95926",
    "#c14d1f",
    "#a64219",
    "#8b3714",
    "#702c10",
    "#55210c",
)
"""Second sequential hue, for when two sequential contexts share a screen."""

# Ordinal ramps must clear 2:1 against the surface at the end nearest it, so
# they start further in than the sequential ramp does.
ORDINAL_START = {"light": 3, "dark": 10}
"""Index into a sequential ramp where an *ordinal* (discrete) ramp may begin."""

DIVERGING: dict[Mode, dict[str, object]] = {
    "light": {
        "low": SEQUENTIAL_BLUE,
        "high": (
            "#fbd9d9",
            "#f6bcbc",
            "#f09f9f",
            "#ea8382",
            "#e34948",
            "#d03b3b",
            "#b23131",
            "#932828",
            "#751f1f",
        ),
        "mid": "#f0efec",
    },
    "dark": {
        "low": SEQUENTIAL_BLUE,
        "high": (
            "#fbd9d9",
            "#f6bcbc",
            "#f09f9f",
            "#ea8382",
            "#e66767",
            "#d03b3b",
            "#b23131",
            "#932828",
            "#751f1f",
        ),
        "mid": "#383835",
    },
}
"""Diverging = two hues (blue <-> red) with a **neutral gray** midpoint.

Never a rainbow, and never a hue at the midpoint - the middle must read as
"nothing", which only a neutral does.
"""

STATUS: dict[str, str] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
"""Reserved status colors. Never reused as "series 4", and always shipped with
an icon + label so state is never carried by color alone."""

CHROME: dict[Mode, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "text_muted": "#898781",
        "gridline": "#e1e0d9",
        "axis": "#c3c2b7",
        "delta_good": "#006300",
        "delta_bad": "#d03b3b",
        "border": "rgba(11,11,11,0.10)",
        "deemphasis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "text_muted": "#898781",
        "gridline": "#2c2c2a",
        "axis": "#383835",
        "delta_good": "#0ca30c",
        "delta_bad": "#e66767",
        "border": "rgba(255,255,255,0.10)",
        "deemphasis": "#52514e",
    },
}
"""Chart chrome and ink. Text always wears text tokens - never the series color."""

SURFACES: dict[Mode, str] = {"light": "#fcfcfb", "dark": "#1a1a19"}

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", sans-serif'


# --------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------


def chrome(mode: Mode = "light") -> dict[str, str]:
    """Chrome/ink tokens for *mode*."""
    _check_mode(mode)
    return dict(CHROME[mode])


def status_color(role: str, mode: Mode = "light") -> str:
    """Look up a reserved status color. *mode* is accepted for symmetry; the
    status palette is deliberately mode-invariant."""
    _check_mode(mode)
    try:
        return STATUS[role]
    except KeyError:
        raise ValueError(
            f"unknown status role {role!r}; expected one of {sorted(STATUS)}"
        ) from None


def categorical(n: int, mode: Mode = "light", pairs: PairList = "adjacent") -> list[str]:
    """Return the first *n* categorical slots, in fixed order.

    Parameters
    ----------
    n:
        How many series need a color.
    mode:
        ``"light"`` or ``"dark"``.
    pairs:
        ``"adjacent"`` for forms where only neighbours touch (bars, stacks,
        lines); ``"all"`` for forms where any two series can end up adjacent
        (scatter, bubble, choropleth, small multiples).

    Raises
    ------
    ValueError
        If *n* exceeds the cap for the pairlist. This is deliberate: the fix
        for "too many series" is folding the tail into "Other", faceting, or
        composite encoding - **never** generating a 9th hue, which is
        indistinguishable from an existing one under CVD.
    """
    _check_mode(mode)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    cap = ALL_PAIRS_SERIES_CAP if pairs == "all" else ADJACENT_SERIES_CAP
    if n > cap:
        raise ValueError(
            f"{n} series exceeds the {pairs}-pairs cap of {cap}. Fold the tail into "
            f"'Other', facet into small multiples, or use composite encoding "
            f"(hue x shape) - do not generate additional hues."
        )
    return list(CATEGORICAL[mode][:n])


def sequential_ramp(n: int, mode: Mode = "light", hue: str = "blue", ordinal: bool = False) -> list[str]:
    """Sample *n* evenly spaced steps from a single-hue ramp, light -> dark.

    With ``ordinal=True`` the ramp is clipped so the step nearest the surface
    still clears 2:1 contrast - discrete ordered marks (funnel stages, tiers)
    must each stay visible, whereas a continuous sequential scale is allowed to
    recede toward the surface at "near zero".

    In dark mode the ramp is reversed so that "more" is still the step furthest
    from the surface.
    """
    _check_mode(mode)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    base = {"blue": SEQUENTIAL_BLUE, "orange": SEQUENTIAL_ORANGE}.get(hue)
    if base is None:
        raise ValueError(f"unknown sequential hue {hue!r}; expected 'blue' or 'orange'")

    ramp = list(base)
    if ordinal:
        ramp = ramp[ORDINAL_START["light"] :] if mode == "light" else ramp[: ORDINAL_START["dark"] + 1]
    if mode == "dark":
        ramp = list(reversed(ramp))

    if n == 1:
        return [ramp[len(ramp) // 2]]
    step = (len(ramp) - 1) / (n - 1)
    return [ramp[int(round(i * step))] for i in range(n)]


def diverging_ramp(n: int, mode: Mode = "light") -> list[str]:
    """Sample *n* steps across the diverging scale, low -> mid -> high.

    Odd *n* puts the neutral midpoint at the center; even *n* omits it, giving
    equal step counts per arm.
    """
    _check_mode(mode)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    spec = DIVERGING[mode]
    low: Sequence[str] = spec["low"]  # type: ignore[assignment]
    high: Sequence[str] = spec["high"]  # type: ignore[assignment]
    mid: str = spec["mid"]  # type: ignore[assignment]

    if n == 1:
        return [mid]

    has_mid = n % 2 == 1
    per_arm = (n - 1) // 2 if has_mid else n // 2

    def arm(ramp: Sequence[str], count: int, dark_first: bool) -> list[str]:
        # Take the darkest `count` steps of the arm, ordered outside -> inside.
        picked = [ramp[len(ramp) - 1 - int(round(i * (len(ramp) - 1) / max(count, 1) * 0.6))] for i in range(count)]
        return picked if dark_first else list(reversed(picked))

    out = arm(low, per_arm, dark_first=True)
    if has_mid:
        out.append(mid)
    out += list(reversed(arm(high, per_arm, dark_first=True)))
    return out


def emphasis_pair(mode: Mode = "light") -> tuple[str, str]:
    """``(accent, deemphasis)`` for the emphasis form - one series in the accent
    hue, every other series in the de-emphasis gray."""
    _check_mode(mode)
    return CATEGORICAL[mode][0], CHROME[mode]["deemphasis"]


def _check_mode(mode: str) -> None:
    if mode not in ("light", "dark"):
        raise ValueError(f"mode must be 'light' or 'dark', got {mode!r}")


# --------------------------------------------------------------------------
# The validator
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """One row of the validator report."""

    check: str
    severity: Severity
    subject: str
    measured: float | None
    threshold: float | None
    message: str

    @property
    def ok(self) -> bool:
        return self.severity != "FAIL"

    def as_dict(self) -> dict[str, object]:
        return {
            "check": self.check,
            "severity": self.severity,
            "subject": self.subject,
            "measured": None if self.measured is None else round(self.measured, 2),
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass(frozen=True)
class PaletteReport:
    """Aggregate result of running the six checks over a palette."""

    palette: tuple[str, ...]
    mode: Mode
    surface: str
    pairs: PairList
    results: tuple[CheckResult, ...] = field(default_factory=tuple)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.severity == "FAIL"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.severity == "WARN"]

    @property
    def passed(self) -> bool:
        """True when nothing FAILs. WARNs do not block, but they carry
        obligations (secondary encoding / the relief rule)."""
        return not self.failures

    @property
    def requires_secondary_encoding(self) -> bool:
        """A CVD pair landed in the 6-8 WARN band: direct labels, texture, or
        shape must carry identity alongside hue."""
        return any(r.check == "cvd_separation" and r.severity == "WARN" for r in self.results)

    @property
    def requires_relief(self) -> bool:
        """A slot fell below 3:1 on the surface: visible direct labels or a
        table view are obligatory."""
        return any(r.check == "surface_contrast" and r.severity == "WARN" for r in self.results)

    def as_dict(self) -> dict[str, object]:
        return {
            "palette": list(self.palette),
            "mode": self.mode,
            "surface": self.surface,
            "pairs": self.pairs,
            "passed": self.passed,
            "requires_secondary_encoding": self.requires_secondary_encoding,
            "requires_relief": self.requires_relief,
            "results": [r.as_dict() for r in self.results],
        }

    def format_table(self) -> str:
        """Render the report as a fixed-width table for terminal output."""
        header = f"palette={','.join(self.palette)}  mode={self.mode}  surface={self.surface}  pairs={self.pairs}"
        rows = [header, "-" * max(len(header), 78)]
        rows.append(f"{'CHECK':<20}{'RESULT':<8}{'SUBJECT':<22}{'MEASURED':>10}  {'LIMIT':>7}")
        for r in self.results:
            measured = "-" if r.measured is None else f"{r.measured:.2f}"
            limit = "-" if r.threshold is None else f"{r.threshold:.2f}"
            rows.append(
                f"{r.check:<20}{r.severity:<8}{r.subject:<22}{measured:>10}  {limit:>7}"
            )
        rows.append("-" * max(len(header), 78))
        verdict = "PASS" if self.passed else f"FAIL ({len(self.failures)} blocking)"
        rows.append(f"verdict: {verdict}; {len(self.warnings)} warning(s)")
        if self.requires_secondary_encoding:
            rows.append("  -> secondary encoding REQUIRED (direct labels / texture / shape)")
        if self.requires_relief:
            rows.append("  -> relief rule REQUIRED (visible direct labels or a table view)")
        for r in self.failures:
            rows.append(f"  FAIL {r.check} [{r.subject}]: {r.message}")
        return "\n".join(rows)


def _pairlist(n: int, pairs: PairList) -> list[tuple[int, int]]:
    if pairs == "all":
        return list(combinations(range(n), 2))
    if pairs == "adjacent":
        return [(i, i + 1) for i in range(n - 1)]
    raise ValueError(f"pairs must be 'adjacent' or 'all', got {pairs!r}")


def validate_palette(
    palette: Sequence[str],
    mode: Mode = "light",
    surface: str | None = None,
    pairs: PairList = "adjacent",
    cvd_kinds: Iterable[CVDKind] = GATED_CVD_KINDS,
) -> PaletteReport:
    """Run the six checks over *palette* and return a :class:`PaletteReport`.

    This is the gate every categorical palette passes before it reaches a
    chart. It never mutates the palette and never "fixes" anything - it
    reports, and the caller decides.

    Parameters
    ----------
    palette:
        Hex colors, in the order they will be assigned to series.
    mode:
        Which mode's lightness band and default surface to check against.
    surface:
        Chart surface hex. Defaults to the mode's reference surface. **Always
        pass your own when you swap in your own palette** - contrast results
        are only meaningful against the surface the chart actually renders on.
    pairs:
        ``"adjacent"`` or ``"all"`` (see module docstring).
    cvd_kinds:
        Which dichromacies the gate considers. Defaults to protan + deutan;
        tritan is always measured and reported alongside, but never blocks.

    Raises
    ------
    ValueError
        If any color is not a six-digit hex. Parsing is guarded at this
        boundary on purpose: unguarded, a malformed string propagates through
        every check and the run fails **open**, reporting a pass it never
        earned.
    """
    _check_mode(mode)
    if not palette:
        raise ValueError("palette must contain at least one color")
    for c in palette:
        if not is_hex_color(c):
            raise ValueError(
                f"not a six-digit hex color: {c!r} (three-digit shorthand is not "
                f"accepted at the validator boundary)"
            )
    normalized = tuple(to_hex(parse_hex(strip_ws(c))) for c in palette)
    surf = surface if surface is not None else SURFACES[mode]
    if not is_hex_color(surf):
        raise ValueError(f"surface is not a six-digit hex color: {surf!r}")
    surf = to_hex(parse_hex(strip_ws(surf)))
    lo, hi = LIGHTNESS_BAND[mode]
    results: list[CheckResult] = []

    # 1. lightness band
    for i, c in enumerate(normalized):
        L = hex_to_oklch(c).L
        inside = lo <= L <= hi
        results.append(
            CheckResult(
                check="lightness_band",
                severity="PASS" if inside else "FAIL",
                subject=f"slot {i + 1} {c}",
                measured=L,
                threshold=None,
                message=(
                    f"OKLCh L {L:.3f} inside the {mode} band [{lo}, {hi}]"
                    if inside
                    else f"OKLCh L {L:.3f} outside the {mode} band [{lo}, {hi}] - "
                    f"re-step this slot on its own ramp"
                ),
            )
        )

    # 2. chroma floor
    for i, c in enumerate(normalized):
        C = hex_to_oklch(c).C
        ok = C >= CHROMA_FLOOR
        results.append(
            CheckResult(
                check="chroma_floor",
                severity="PASS" if ok else "FAIL",
                subject=f"slot {i + 1} {c}",
                measured=C,
                threshold=CHROMA_FLOOR,
                message=(
                    f"chroma {C:.3f} carries a hue"
                    if ok
                    else f"chroma {C:.3f} reads as a neutral, not an identity hue"
                ),
            )
        )

    # 3 & 4. pairwise separation, simulated and unsimulated
    for i, j in _pairlist(len(normalized), pairs):
        a, b = normalized[i], normalized[j]
        label = f"{i + 1}x{j + 1}"

        worst_d = None
        worst_kind: CVDKind | None = None
        for kind in cvd_kinds:
            d = delta_e(a, b, kind)
            if worst_d is None or d < worst_d:
                worst_d, worst_kind = d, kind
        if worst_d is None or worst_kind is None:
            raise ValueError("cvd_kinds must not be empty")

        d_tritan = delta_e(a, b, "tritan")
        tritan_note = f"; tritan {d_tritan:.1f} (reported, not gated)"

        if worst_d >= CVD_TARGET:
            sev: Severity = "PASS"
            msg = f"separates under {worst_kind} (worst case){tritan_note}"
        elif worst_d >= CVD_FLOOR:
            sev = "WARN"
            msg = (
                f"in the {CVD_FLOOR}-{CVD_TARGET} warn band under {worst_kind}; legal ONLY with "
                f"secondary encoding (direct labels / texture / shape){tritan_note}"
            )
        else:
            sev = "FAIL"
            msg = f"indistinguishable under {worst_kind}; re-step or cut a series{tritan_note}"
        results.append(
            CheckResult("cvd_separation", sev, label, worst_d, CVD_TARGET, msg)
        )

        d_normal = delta_e(a, b)
        n_ok = d_normal >= NORMAL_VISION_FLOOR
        results.append(
            CheckResult(
                check="normal_vision",
                severity="PASS" if n_ok else "FAIL",
                subject=label,
                measured=d_normal,
                threshold=NORMAL_VISION_FLOOR,
                message=(
                    "separates for full-color readers"
                    if n_ok
                    else "full-color readers cannot separate this pair - hard fail, "
                    "secondary encoding does not excuse it; re-step or cut a series"
                ),
            )
        )

    # 5. surface contrast
    for i, c in enumerate(normalized):
        ratio = contrast_ratio(c, surf)
        ok = ratio >= CONTRAST_TARGET
        results.append(
            CheckResult(
                check="surface_contrast",
                severity="PASS" if ok else "WARN",
                subject=f"slot {i + 1} {c}",
                measured=ratio,
                threshold=CONTRAST_TARGET,
                message=(
                    f"{ratio:.2f}:1 against {surf}"
                    if ok
                    else f"{ratio:.2f}:1 against {surf} is below {CONTRAST_TARGET}:1 - "
                    f"the relief rule applies (visible direct labels or a table view)"
                ),
            )
        )

    # 6. distinctness
    seen: dict[str, int] = {}
    for i, c in enumerate(normalized):
        if c in seen:
            results.append(
                CheckResult(
                    "distinctness",
                    "FAIL",
                    f"{seen[c] + 1}x{i + 1}",
                    0.0,
                    None,
                    f"slots {seen[c] + 1} and {i + 1} are the same color ({c})",
                )
            )
        else:
            seen[c] = i
    if len(seen) == len(normalized):
        results.append(
            CheckResult("distinctness", "PASS", "all slots", None, None, "no duplicate slots")
        )

    return PaletteReport(
        palette=normalized, mode=mode, surface=surf, pairs=pairs, results=tuple(results)
    )
