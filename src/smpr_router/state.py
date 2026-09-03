from __future__ import annotations

from dataclasses import dataclass, field

from .circuit_dag import CircuitDAG
from .hardware import HardwareGraph


@dataclass
class RoutingState:
    """Mutable logical-to-physical routing state."""

    logical_to_physical: list[int]
    physical_to_logical: list[int | None]
    remaining_predecessors: list[int]
    executed_gates: set[int] = field(default_factory=set)
    physical_operations: list[tuple] = field(default_factory=list)
    physical_depth: list[int] = field(default_factory=list)

    def __deepcopy__(self, memo: dict[int, object]) -> "RoutingState":
        existing = memo.get(id(self))
        if existing is not None:
            return existing  # type: ignore[return-value]
        cloned = RoutingState(
            logical_to_physical=self.logical_to_physical.copy(),
            physical_to_logical=self.physical_to_logical.copy(),
            remaining_predecessors=self.remaining_predecessors.copy(),
            executed_gates=self.executed_gates.copy(),
            physical_operations=self.physical_operations.copy(),
            physical_depth=self.physical_depth.copy(),
        )
        memo[id(self)] = cloned
        return cloned

    @classmethod
    def initial(
        cls,
        dag: CircuitDAG,
        hardware: HardwareGraph,
        mapping: list[int],
    ) -> "RoutingState":
        if len(mapping) != dag.num_logical_qubits:
            raise ValueError("mapping length does not match the logical circuit")
        if len(set(mapping)) != len(mapping):
            raise ValueError("logical-to-physical mapping must be injective")
        if any(not 0 <= physical < hardware.num_qubits for physical in mapping):
            raise ValueError("mapping uses a vertex outside the hardware")

        physical_to_logical: list[int | None] = [
            None for _ in range(hardware.num_qubits)
        ]
        for logical, physical in enumerate(mapping):
            physical_to_logical[physical] = logical
        state = cls(
            logical_to_physical=list(mapping),
            physical_to_logical=physical_to_logical,
            remaining_predecessors=[
                len(dag.predecessors[gate.gate_id]) for gate in dag.gates
            ],
            physical_depth=[0 for _ in range(hardware.num_qubits)],
        )
        state.assert_valid(dag, hardware)
        return state

    def current_depth(self) -> int:
        return max(self.physical_depth, default=0)

    def assert_valid(self, dag: CircuitDAG, hardware: HardwareGraph) -> None:
        occupied: set[int] = set()
        for logical, physical in enumerate(self.logical_to_physical):
            if physical in occupied:
                raise AssertionError("logical-to-physical mapping is not injective")
            occupied.add(physical)
            if self.physical_to_logical[physical] != logical:
                raise AssertionError("forward and inverse mappings disagree")

        for physical, logical in enumerate(self.physical_to_logical):
            if (
                logical is not None
                and self.logical_to_physical[logical] != physical
            ):
                raise AssertionError("inverse and forward mappings disagree")

        for gate_id in self.executed_gates:
            if not dag.predecessors[gate_id].issubset(self.executed_gates):
                raise AssertionError("an executed gate violates DAG order")

        for operation in self.physical_operations:
            if operation[0] in {"cx", "swap"} and not hardware.adjacent(
                operation[1], operation[2]
            ):
                raise AssertionError(f"illegal two-qubit operation: {operation}")

    def append_single_qubit_gate(
        self,
        name: str,
        physical_qubit: int,
        source_gate_id: int,
    ) -> None:
        self.physical_depth[physical_qubit] += 1
        self.physical_operations.append((name, physical_qubit, source_gate_id))

    def append_two_qubit_gate(
        self,
        name: str,
        u: int,
        v: int,
        source_gate_id: int | None,
        hardware: HardwareGraph,
        duration: int = 1,
    ) -> None:
        if not hardware.adjacent(u, v):
            raise ValueError(f"hardware vertices {u} and {v} are not adjacent")
        new_depth = max(self.physical_depth[u], self.physical_depth[v]) + duration
        self.physical_depth[u] = new_depth
        self.physical_depth[v] = new_depth
        self.physical_operations.append((name, u, v, source_gate_id))

    def apply_swap(self, u: int, v: int, hardware: HardwareGraph) -> None:
        self.append_two_qubit_gate(
            "swap",
            u,
            v,
            source_gate_id=None,
            hardware=hardware,
            duration=3,
        )
        logical_u = self.physical_to_logical[u]
        logical_v = self.physical_to_logical[v]
        self.physical_to_logical[u] = logical_v
        self.physical_to_logical[v] = logical_u
        if logical_u is not None:
            self.logical_to_physical[logical_u] = v
        if logical_v is not None:
            self.logical_to_physical[logical_v] = u
