from __future__ import annotations

import math
from typing import Any, Mapping

from v1_simulation.analysis import AnalysisResult


def score_analysis_result(result: AnalysisResult) -> dict[str, Any]:
    """Extract sweep-friendly metrics and a deterministic quality score."""

    diagnostics = result.diagnostics
    summary = diagnostics.get("metrics_summary", {})
    if not isinstance(summary, Mapping):
        summary = {}

    score = quality_score(summary, diagnostics)
    communities = result.communities
    return {
        "analysis_status": result.status,
        "score": score,
        "selected_neurons": int(result.selected_indices.size),
        "n_ensembles": _as_int(summary.get("n_ensembles")),
        "classified_fraction": _as_float(summary.get("classified_fraction")),
        "active_fraction": _as_float(diagnostics.get("active_fraction")),
        "silent_fraction": _as_float(diagnostics.get("silent_fraction")),
        "rate_mean": _as_float(diagnostics.get("rate_mean")),
        "rate_max": _as_float(diagnostics.get("rate_max")),
        "osi_mean": _as_float(summary.get("osi_mean")),
        "osi_median": _as_float(summary.get("osi_median")),
        "community_classified_neurons": None if communities is None else communities.classified_neurons,
    }


def quality_score(
    summary: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None = None,
) -> float:
    """Heuristic score for ranking sweep runs.

    The score rewards active neurons, classified ensemble fraction, OSI, and multiple
    detected ensembles. It is intentionally simple and transparent so users can replace
    it with a project-specific scientific objective when needed.
    """

    diagnostics = diagnostics or {}
    classified = _finite_or_zero(summary.get("classified_fraction"))
    active = _finite_or_zero(diagnostics.get("active_fraction"))
    osi = _finite_or_zero(summary.get("osi_mean"))
    n_ensembles = max(0, int(_finite_or_zero(summary.get("n_ensembles"))))
    silent = _finite_or_zero(diagnostics.get("silent_fraction"))
    dominance = _finite_or_zero(diagnostics.get("top1_activity_fraction"))

    diversity = math.log1p(n_ensembles)
    penalty = max(0.0, 1.0 - 0.5 * silent - 0.25 * dominance)
    return float(classified * active * (1.0 + osi) * (1.0 + diversity) * penalty)


def _finite_or_zero(value: Any) -> float:
    number = _as_float(value)
    return 0.0 if number is None else number


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return None if number is None else int(number)


__all__ = ["quality_score", "score_analysis_result"]
