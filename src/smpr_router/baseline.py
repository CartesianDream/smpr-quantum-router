from __future__ import annotations

from .circuit_dag import CircuitDAG
from .drain import blocked_front_layer, drain
from .hardware import HardwareGraph
from .state import RoutingState


def run_shortest_path_baseline(
    dag: CircuitDAG,
    hardware: HardwareGraph,
    initial_mapping: list[int],
    *,
    max_swaps: int = 1000,
) -> RoutingState:
    """Route by moving the first blocked gate along a deterministic shortest path."""

    state = RoutingState.initial(dag, hardware, initial_mapping)
    swaps = 0
    while len(state.executed_gates) < len(dag.gates):
        drain(state, dag, hardware)
        if len(state.executed_gates) == len(dag.gates):
            break
        front = blocked_front_layer(state, dag)
        if not front:
            raise RuntimeError("unfinished circuit has an empty front layer")
        gate = dag.gates[front[0]]
        u = state.logical_to_physical[gate.qubits[0]]
        v = state.logical_to_physical[gate.qubits[1]]
        path = hardware.shortest_path(u, v)
        if len(path) < 2:
            raise RuntimeError("blocked gate produced a degenerate shortest path")
        state.apply_swap(path[0], path[1], hardware)
        swaps += 1
        if swaps > max_swaps:
            raise RuntimeError("maximum SWAP count exceeded")
    state.assert_valid(dag, hardware)
    return state

