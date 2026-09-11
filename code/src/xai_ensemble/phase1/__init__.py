"""Explanation generation primitives and the explainer-compatibility pilot."""

from ._compat_support import attribution_to_patch_scores, scores_to_ranks
from .explainers import EXPLAINER_SPECS, ExplainerSpec, build_explainer

__all__ = [
    "EXPLAINER_SPECS",
    "ExplainerSpec",
    "attribution_to_patch_scores",
    "build_explainer",
    "scores_to_ranks",
]
