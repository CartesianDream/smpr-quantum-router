from __future__ import annotations

from .circuit_dag import CircuitDAG
from .hardware import HardwareGraph
from .state import RoutingState


def drain(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    *,
    validate: bool = True,
) -> list[int]:
    """Execute the closure of dependency-ready and topology-valid gates."""

    executed_now: list[int] = []
    made_progress = True
    while made_progress:
        made_progress = False
        for gate in dag.gates:
            gate_id = gate.gate_id
            if gate_id in state.executed_gates:
                continue
            if state.remaining_predecessors[gate_id] != 0:
                continue

            if gate.is_single_qubit:
                physical = state.logical_to_physical[gate.qubits[0]]
                state.append_single_qubit_gate(gate.name, physical, gate_id)
            elif gate.is_two_qubit:
                u = state.logical_to_physical[gate.qubits[0]]
                v = state.logical_to_physical[gate.qubits[1]]
                if not hardware.adjacent(u, v):
                    continue
                state.append_two_qubit_gate(
                    gate.name,
                    u,
                    v,
                    source_gate_id=gate_id,
                    hardware=hardware,
                    duration=1,
                )
            else:  # Defensive: CircuitDAG already rejects other arities.
                raise ValueError("only one- and two-qubit gates are supported")

            state.executed_gates.add(gate_id)
            executed_now.append(gate_id)
            for successor in dag.successors[gate_id]:
                state.remaining_predecessors[successor] -= 1
                if state.remaining_predecessors[successor] < 0:
                    raise AssertionError("remaining predecessor count became negative")
            made_progress = True

    if validate:
        state.assert_valid(dag, hardware)
    return executed_now


def blocked_front_layer(state: RoutingState, dag: CircuitDAG) -> list[int]:
    """Return dependency-ready two-qubit gates blocked by connectivity."""

    return [
        gate.gate_id
        for gate in dag.gates
        if gate.gate_id not in state.executed_gates
        and state.remaining_predecessors[gate.gate_id] == 0
        and gate.is_two_qubit
    ]
