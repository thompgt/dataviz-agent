"""dataviz-agent - natural language in, an accessible validated chart out.

The package is the deterministic toolkit an LLM agent drives. The split is
deliberate:

* **The agent** (see ``AGENT.md`` / ``skills/dataviz-agent/SKILL.md``) handles
  what language models are good at: reading an ambiguous question, mapping
  "revenue" onto the column actually called ``net_rev_usd``, and deciding what
  the reader is really asking.
* **This package** handles what code should never delegate: profiling the data,
  executing the aggregation, choosing the form from the profile, validating the
  palette, and rendering. None of it guesses.

The seam between them is :class:`~dataviz_agent.spec.ChartSpec` - a validated,
serializable description of exactly one chart. The agent can emit one directly;
:func:`~dataviz_agent.nlq.parse_query` will also derive one from plain English
without any model in the loop.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .color import contrast_ratio, delta_e, hex_to_oklch, simulate_cvd_hex
from .errors import (
    AmbiguousQueryError,
    DataVizAgentError,
    IngestError,
    PaletteError,
    RenderError,
    SpecError,
    UnsupportedChartError,
)
from .palette import (
    CATEGORICAL,
    PaletteReport,
    categorical,
    chrome,
    diverging_ramp,
    sequential_ramp,
    validate_palette,
)

__all__ = [
    "__version__",
    # errors
    "DataVizAgentError",
    "IngestError",
    "SpecError",
    "PaletteError",
    "RenderError",
    "AmbiguousQueryError",
    "UnsupportedChartError",
    # color
    "delta_e",
    "contrast_ratio",
    "hex_to_oklch",
    "simulate_cvd_hex",
    # palette
    "CATEGORICAL",
    "PaletteReport",
    "validate_palette",
    "categorical",
    "sequential_ramp",
    "diverging_ramp",
    "chrome",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy-import shim
    """Lazily expose the heavier layers.

    ``profile``, ``nlq``, ``recommend``, ``transform`` and ``render`` pull in
    pandas and (optionally) a plotting backend. Importing them eagerly would
    make ``import dataviz_agent`` slow for callers who only want the color
    math, so they resolve on first attribute access instead.
    """
    lazy = {
        "ChartSpec": "spec",
        "Encoding": "spec",
        "profile_frame": "profile",
        "DataProfile": "profile",
        "parse_query": "nlq",
        "recommend": "recommend",
        "apply_spec": "transform",
        "render": "render",
        "visualize": "pipeline",
    }
    module = lazy.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f".{module}", __name__)
    return getattr(mod, name) if name != module else mod
