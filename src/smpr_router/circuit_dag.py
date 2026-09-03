from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Gate:
    """A logical one- or two-qubit gate."""

    gate_id: int
    name: str
    qubits: tuple[int, ...]

    @property
    def is_single_qubit(self) -> bool:
        return len(self.qubits) == 1

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2


class CircuitDAG:
    """Dependency DAG induced by per-logical-qubit gate order."""

    def __init__(self, num_logical_qubits: int, gates: list[Gate]) -> None:
        if num_logical_qubits <= 0:
            raise ValueError("num_logical_qubits must be positive")
        if [gate.gate_id for gate in gates] != list(range(len(gates))):
            raise ValueError("gate_id values must be contiguous from zero")
        for gate in gates:
            if len(gate.qubits) not in (1, 2):
                raise ValueError("only one- and two-qubit gates are supported")
            if any(not 0 <= qubit < num_logical_qubits for qubit in gate.qubits):
                raise ValueError(f"invalid logical qubit in gate {gate}")
            if len(set(gate.qubits)) != len(gate.qubits):
                raise ValueError(f"gate repeats a logical qubit: {gate}")

        self.num_logical_qubits = num_logical_qubits
        self.gates = gates
        self.predecessors: list[set[int]] = [set() for _ in gates]
        self.successors: list[set[int]] = [set() for _ in gates]
        self._build_dependencies()

    def _build_dependencies(self) -> None:
        last_gate_on_qubit: list[int | None] = [
            None for _ in range(self.num_logical_qubits)
        ]
        for gate in self.gates:
            predecessors = {
                previous
                for logical_qubit in gate.qubits
                if (previous := last_gate_on_qubit[logical_qubit]) is not None
            }
            self.predecessors[gate.gate_id] = predecessors
            for predecessor in predecessors:
                self.successors[predecessor].add(gate.gate_id)
            for logical_qubit in gate.qubits:
                last_gate_on_qubit[logical_qubit] = gate.gate_id
