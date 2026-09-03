from __future__ import annotations

import copy
import unittest

from smpr_router import (
    CircuitDAG,
    Gate,
    HardwareGraph,
    RoutingState,
    count_swaps,
    quality,
    run_shortest_path_baseline,
)


class CoreModelTests(unittest.TestCase):
    def test_dag_dependencies_follow_each_logical_wire(self) -> None:
        dag = CircuitDAG(
            3,
            [
                Gate(0, "h", (0,)),
                Gate(1, "cx", (0, 2)),
                Gate(2, "h", (2,)),
            ],
        )
        self.assertEqual(dag.predecessors, [set(), {0}, {1}])
        self.assertEqual(dag.successors, [{1}, {2}, set()])

    def test_shortest_path_is_deterministic(self) -> None:
        hardware = HardwareGraph(4, [(0, 1), (1, 3), (0, 2), (2, 3)])
        self.assertEqual(hardware.shortest_path(0, 3), [0, 1, 3])
        self.assertEqual(hardware.shortest_path(0, 3), [0, 1, 3])

    def test_state_deepcopy_is_independent(self) -> None:
        dag = CircuitDAG(2, [Gate(0, "cx", (0, 1))])
        hardware = HardwareGraph(2, [(0, 1)])
        state = RoutingState.initial(dag, hardware, [0, 1])
        cloned = copy.deepcopy(state)
        cloned.logical_to_physical[0] = 1
        self.assertEqual(state.logical_to_physical, [0, 1])

    def test_baseline_routes_nonlocal_gate(self) -> None:
        dag = CircuitDAG(3, [Gate(0, "cx", (0, 2))])
        hardware = HardwareGraph(3, [(0, 1), (1, 2)])
        state = run_shortest_path_baseline(dag, hardware, [0, 1, 2])
        self.assertEqual(count_swaps(state), 1)
        self.assertEqual(quality(state), (1, 4))
        state.assert_valid(dag, hardware)

    def test_invalid_mapping_is_rejected(self) -> None:
        dag = CircuitDAG(2, [Gate(0, "cx", (0, 1))])
        hardware = HardwareGraph(2, [(0, 1)])
        with self.assertRaises(ValueError):
            RoutingState.initial(dag, hardware, [0, 0])


if __name__ == "__main__":
    unittest.main()

