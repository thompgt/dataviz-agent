"""Color science primitives: sRGB <-> OKLab/OKLCh, CVD simulation, WCAG contrast.

Everything in this module is pure, deterministic, and dependency-free (stdlib
`math` only). It exists so the palette checks in :mod:`dataviz_agent.palette`
are *computed* rather than eyeballed.

Conventions
-----------
* A "hex" is a string like ``"#2a78d6"``; ``#rgb`` shorthand and a missing
  leading ``#`` are both accepted on input, and output is always lowercase
  six-digit with a leading ``#``.
* ``RGB`` triples are floats in ``[0, 1]`` in **gamma-encoded** sRGB.
* ``LinearRGB`` triples are floats in ``[0, 1]`` in **linear-light** sRGB.
* OKLab ``L`` is in ``[0, 1]``; ``a``/``b`` are roughly ``[-0.4, 0.4]``.
* Perceptual distances are reported as **OKLab dE x100** ("dE units" below),
  which is the unit the dataviz gates are written in (>= 8 CVD, >= 15 normal).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

__all__ = [
    "RGB",
    "OKLab",
    "OKLCh",
    "CVDKind",
    "parse_hex",
    "to_hex",
    "srgb_to_linear",
    "linear_to_srgb",
    "hex_to_oklab",
    "oklab_to_oklch",
    "hex_to_oklch",
    "delta_e_oklab",
    "delta_e_hex",
    "relative_luminance",
    "contrast_ratio",
    "simulate_cvd",
    "simulate_cvd_linear",
    "simulate_cvd_hex",
    "delta_e",
    "worst_cvd_delta_e",
    "is_hex_color",
    "split_colors",
    "strip_ws",
    "CVD_KINDS",
    "GATED_CVD_KINDS",
]

CVDKind = Literal["protan", "deutan", "tritan"]
CVD_KINDS: tuple[CVDKind, ...] = ("protan", "deutan", "tritan")

GATED_CVD_KINDS: tuple[CVDKind, ...] = ("protan", "deutan")
"""The dichromacies the separation gate is calibrated against.

Tritan is *reported* rather than gated: tritanopia is rare enough, and its
confusion axis different enough, that calibrating the 8.0 dE target against it
would reject palettes that work in practice. Report it; don't block on it.
"""

_HEX_RE = re.compile(r"^#?(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

# Whitespace normalization for user-supplied color strings. Kept as an explicit
# set rather than relying on `str.strip()` because this must stay in lockstep
# with the JavaScript twin, and JS `trim()` and Python `str.strip()` differ at
# the edges (trim() strips U+FEFF; strip() strips U+001C-U+001F and U+0085).
# This is their intersection: ASCII whitespace plus the Unicode space/separator
# characters both engines strip - which also covers the NBSP/em-space padding
# picked up when hex lists are copy-pasted out of a rendered page.
_WS = (
    " \t\n\v\f\r"          # ASCII whitespace
    "\u00a0\u1680"           # NBSP, Ogham space mark
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
_STRICT_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


def strip_ws(value: str) -> str:
    """Trim the shared JS/Python whitespace set from both ends of *value*."""
    return value.strip(_WS)


def is_hex_color(value: str) -> bool:
    """Strict six-digit hex test used at the validator's input boundary.

    Deliberately narrower than :func:`parse_hex`: unguarded parsing propagates
    garbage through every check and makes a run fail *open*, so user-supplied
    palette and surface strings must clear this first. Three-digit shorthand is
    rejected here even though :func:`parse_hex` accepts it internally.
    """
    return isinstance(value, str) and bool(_STRICT_HEX_RE.match(strip_ws(value)))


def split_colors(raw: str) -> list[str]:
    """Split a comma-separated color list, trimming and dropping empties."""
    return [c for c in (strip_ws(p) for p in (raw or "").split(",")) if c]


@dataclass(frozen=True)
class RGB:
    """Gamma-encoded sRGB in [0, 1]."""

    r: float
    g: float
    b: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.r, self.g, self.b)


@dataclass(frozen=True)
class OKLab:
    L: float
    a: float
    b: float


@dataclass(frozen=True)
class OKLCh:
    """OKLab in polar form. ``h`` is degrees in [0, 360)."""

    L: float
    C: float
    h: float


# --------------------------------------------------------------------------
# hex parsing
# --------------------------------------------------------------------------


def parse_hex(value: str) -> RGB:
    """Parse ``#rgb`` / ``#rrggbb`` (``#`` optional) into an :class:`RGB`.

    Raises
    ------
    ValueError
        If *value* is not a syntactically valid hex color.
    """
    if not isinstance(value, str) or not _HEX_RE.match(value.strip()):
        raise ValueError(f"not a valid hex color: {value!r}")
    h = value.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return RGB(int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def to_hex(rgb: RGB) -> str:
    """Render an :class:`RGB` as ``#rrggbb``, clamping out-of-gamut channels."""

    def chan(x: float) -> int:
        return max(0, min(255, int(round(x * 255))))

    return f"#{chan(rgb.r):02x}{chan(rgb.g):02x}{chan(rgb.b):02x}"


# --------------------------------------------------------------------------
# transfer function
# --------------------------------------------------------------------------


def srgb_to_linear(c: float) -> float:
    """sRGB electro-optical transfer function (gamma-encoded -> linear-light)."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    """Inverse of :func:`srgb_to_linear`, clamped to [0, 1]."""
    c = max(0.0, min(1.0, c))
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def _to_linear_triple(rgb: RGB) -> tuple[float, float, float]:
    return (srgb_to_linear(rgb.r), srgb_to_linear(rgb.g), srgb_to_linear(rgb.b))


def _from_linear_triple(t: Sequence[float]) -> RGB:
    return RGB(linear_to_srgb(t[0]), linear_to_srgb(t[1]), linear_to_srgb(t[2]))


# --------------------------------------------------------------------------
# OKLab  (Bjorn Ottosson, 2020)
# --------------------------------------------------------------------------


def linear_rgb_to_oklab(r: float, g: float, b: float) -> OKLab:
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = math.copysign(abs(l) ** (1 / 3), l)
    m_ = math.copysign(abs(m) ** (1 / 3), m)
    s_ = math.copysign(abs(s) ** (1 / 3), s)

    return OKLab(
        L=0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        a=1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        b=0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def hex_to_oklab(value: str) -> OKLab:
    """Convert a hex color straight to OKLab."""
    return linear_rgb_to_oklab(*_to_linear_triple(parse_hex(value)))


def oklab_to_oklch(lab: OKLab) -> OKLCh:
    """Convert OKLab to polar OKLCh. Hue of a neutral is reported as 0."""
    c = math.hypot(lab.a, lab.b)
    h = math.degrees(math.atan2(lab.b, lab.a)) % 360.0 if c > 1e-9 else 0.0
    return OKLCh(L=lab.L, C=c, h=h)


def hex_to_oklch(value: str) -> OKLCh:
    return oklab_to_oklch(hex_to_oklab(value))


def delta_e_oklab(a: OKLab, b: OKLab) -> float:
    """Perceptual distance in **dE units** (Euclidean OKLab distance x100)."""
    return 100.0 * math.sqrt((a.L - b.L) ** 2 + (a.a - b.a) ** 2 + (a.b - b.b) ** 2)


def delta_e_hex(a: str, b: str) -> float:
    """Perceptual distance between two hex colors, in dE units."""
    return delta_e_oklab(hex_to_oklab(a), hex_to_oklab(b))


# --------------------------------------------------------------------------
# WCAG contrast
# --------------------------------------------------------------------------


def relative_luminance(value: str) -> float:
    """WCAG 2.x relative luminance of a hex color, in [0, 1]."""
    r, g, b = _to_linear_triple(parse_hex(value))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two hex colors. Symmetric; in [1, 21]."""
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# --------------------------------------------------------------------------
# CVD simulation  (Machado, Oliveira & Fernandes 2009, severity 1.0)
# --------------------------------------------------------------------------

# These transforms act directly on **linear RGB** - there is no detour through
# LMS and no gamma round-trip. The simulation model is part of the standard the
# separation thresholds are written against, not an implementation detail:
# swapping in e.g. Vienot-1999 moves borderline pairs by several dE and would
# require recalibrating CVD_TARGET / CVD_FLOOR. Do not substitute one casually.
_MACHADO: dict[str, tuple[tuple[float, float, float], ...]] = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def _matmul(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> tuple[float, float, float]:
    return tuple(sum(m * v for m, v in zip(row, vec)) for row in mat)  # type: ignore[return-value]


def simulate_cvd_linear(
    lin: Sequence[float], kind: CVDKind
) -> tuple[float, float, float]:
    """Simulate *lin* (a linear-RGB triple) as seen with the given dichromacy.

    Channels are clamped to [0, 1]; the result stays in linear light so callers
    can go straight to OKLab without an 8-bit round-trip through hex.
    """
    if kind not in _MACHADO:
        raise ValueError(f"unknown CVD kind: {kind!r}; expected one of {CVD_KINDS}")
    out = _matmul(_MACHADO[kind], lin)
    return tuple(max(0.0, min(1.0, c)) for c in out)  # type: ignore[return-value]


def simulate_cvd(rgb: RGB, kind: CVDKind) -> RGB:
    """Simulate how *rgb* appears to a dichromat of the given *kind*."""
    return _from_linear_triple(simulate_cvd_linear(_to_linear_triple(rgb), kind))


def simulate_cvd_hex(value: str, kind: CVDKind) -> str:
    """Hex-in, hex-out wrapper around :func:`simulate_cvd`.

    Useful for *showing* a simulation. Do not measure with it - quantizing to
    8 bits per channel and back perturbs dE. Use :func:`delta_e` instead, which
    stays in linear light.
    """
    return to_hex(simulate_cvd(parse_hex(value), kind))


def delta_e(a: str, b: str, kind: CVDKind | None = None) -> float:
    """Separation between two hex colors in dE units.

    With *kind* set, both colors are passed through that dichromacy simulation
    first. The whole path stays in linear light - no hex round-trip - so the
    result is exact rather than quantized.
    """
    def to_lab(h: str) -> OKLab:
        lin = _to_linear_triple(parse_hex(h))
        if kind is not None:
            lin = simulate_cvd_linear(lin, kind)
        return linear_rgb_to_oklab(*lin)

    return delta_e_oklab(to_lab(a), to_lab(b))


def worst_cvd_delta_e(
    a: str, b: str, kinds: Iterable[CVDKind] = GATED_CVD_KINDS
) -> tuple[float, CVDKind]:
    """Worst-case separation of two colors across *kinds*.

    Returns ``(delta_e, kind)`` for the simulation under which the pair is
    hardest to tell apart. Defaults to the **gated** kinds (protan, deutan) -
    pass ``CVD_KINDS`` explicitly to include tritan.
    """
    worst: tuple[float, CVDKind] | None = None
    for kind in kinds:
        d = delta_e(a, b, kind)
        if worst is None or d < worst[0]:
            worst = (d, kind)
    if worst is None:
        raise ValueError("kinds must not be empty")
    return worst
