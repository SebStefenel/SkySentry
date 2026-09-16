"""Evaluation: detection metrics, FP analysis, sim-to-real gap reporting."""

from .metrics import (
    APResult,
    DetectionEvaluator,
    FPAnalysis,
    domain_gap_report,
    plot_pr_curves,
)

__all__ = ["APResult", "DetectionEvaluator", "FPAnalysis", "domain_gap_report", "plot_pr_curves"]
