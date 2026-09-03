"""Public API for the Safe Multi-Layout Portfolio Router."""

from .baseline import run_shortest_path_baseline
from .circuit_dag import CircuitDAG, Gate
from .drain import blocked_front_layer, drain
from .evaluation import added_cx_count, count_swaps, quality
from .hardware import HardwareGraph
from .selection import SelectionResult, select_safe_candidate
from .state import RoutingState

__all__ = [
    "CircuitDAG",
    "Gate",
    "HardwareGraph",
    "RoutingState",
    "SelectionResult",
    "added_cx_count",
    "blocked_front_layer",
    "count_swaps",
    "drain",
    "quality",
    "run_shortest_path_baseline",
    "select_safe_candidate",
]

__version__ = "1.0.0rc3"
