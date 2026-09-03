from __future__ import annotations

from .circuit_dag import CircuitDAG
from .state import RoutingState


def count_swaps(state: RoutingState) -> int:
    return sum(operation[0] == "swap" for operation in state.physical_operations)


def count_logical_two_qubit_gates(dag: CircuitDAG) -> int:
    return sum(gate.is_two_qubit for gate in dag.gates)


def added_cx_count(state: RoutingState) -> int:
    """Return the standard three-CX cost of abstract SWAP operations."""

    return 3 * count_swaps(state)


def quality(state: RoutingState) -> tuple[int, int]:
    """Frozen lexicographic quality: (SWAP count, weighted depth)."""

    return count_swaps(state), state.current_depth()
