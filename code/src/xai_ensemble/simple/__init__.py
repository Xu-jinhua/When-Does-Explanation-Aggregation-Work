"""Small, paper-first two-stage explanation experiment pipeline.

This package is deliberately independent of the older pilot/formal DAG.  Its
Phase 1 stores full signed attribution maps; its Phase 2 is the only place
where attribution maps are converted to patch scores and ranks.
"""

from .config import SimpleExperiment, load_experiment

__all__ = ["SimpleExperiment", "load_experiment"]
