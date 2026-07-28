"""Exception hierarchy.

Every error the package raises deliberately descends from
:class:`DataVizAgentError`, so an agent driving the toolkit can catch one type
and still tell the failure modes apart. Messages are written to be *actionable
by the agent*: they name the offending column, list the valid alternatives, and
say what to do next - because the reader of a traceback here is usually a model
deciding how to retry, not a human at a REPL.
"""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = [
    "DataVizAgentError",
    "IngestError",
    "SpecError",
    "PaletteError",
    "RenderError",
    "AmbiguousQueryError",
    "UnsupportedChartError",
    "did_you_mean",
]


class DataVizAgentError(Exception):
    """Base class for every error this package raises."""


class IngestError(DataVizAgentError):
    """A data source could not be read, parsed, or typed."""


class SpecError(DataVizAgentError):
    """A :class:`~dataviz_agent.spec.ChartSpec` is malformed or does not match
    the data it was pointed at."""


class PaletteError(DataVizAgentError):
    """A palette failed validation, or more series were requested than the
    palette's cap allows."""


class RenderError(DataVizAgentError):
    """A renderer could not produce output - usually a missing optional
    backend, or a spec the backend does not implement."""


class AmbiguousQueryError(DataVizAgentError):
    """A natural-language query resolved to more than one equally plausible
    reading.

    The ``candidates`` attribute carries the competing interpretations so the
    caller can ask the user rather than silently guessing.
    """

    def __init__(self, message: str, candidates: Sequence[object] = ()) -> None:
        super().__init__(message)
        self.candidates = list(candidates)


class UnsupportedChartError(DataVizAgentError):
    """A chart type was requested that this renderer does not implement.

    Raised in preference to silently substituting a different form - a chart
    the caller did not ask for is worse than a clear failure.
    """


def did_you_mean(name: str, options: Iterable[str], limit: int = 3) -> str:
    """Return a `` (did you mean: a, b)`` suffix, or an empty string.

    Used to turn "no such column" into something an agent can act on in one
    step instead of re-listing the schema.
    """
    import difflib

    matches = difflib.get_close_matches(name, list(options), n=limit, cutoff=0.6)
    if not matches:
        return ""
    return f" (did you mean: {', '.join(matches)}?)"
