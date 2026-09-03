from __future__ import annotations

from collections import deque


class HardwareGraph:
    """Undirected hardware coupling graph with deterministic shortest paths."""

    def __init__(self, num_qubits: int, edges: list[tuple[int, int]]) -> None:
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        self.num_qubits = num_qubits
        self.edges = list(edges)
        self.adjacency: list[set[int]] = [set() for _ in range(num_qubits)]
        self._shortest_path_cache: dict[tuple[int, int], tuple[int, ...]] = {}

        for u, v in edges:
            if not (0 <= u < num_qubits and 0 <= v < num_qubits):
                raise ValueError(f"invalid hardware edge: {(u, v)}")
            if u == v:
                raise ValueError("hardware graph cannot contain self-loops")
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)

    def adjacent(self, u: int, v: int) -> bool:
        return v in self.adjacency[u]

    def shortest_path(self, start: int, goal: int) -> list[int]:
        if not (0 <= start < self.num_qubits and 0 <= goal < self.num_qubits):
            raise ValueError("shortest-path endpoint is outside the hardware")
        cached = self._shortest_path_cache.get((start, goal))
        if cached is not None:
            return list(cached)

        queue: deque[int] = deque([start])
        parent: dict[int, int | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for neighbor in sorted(self.adjacency[current]):
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)

        if goal not in parent:
            raise ValueError(f"hardware vertices {start} and {goal} are disconnected")

        path: list[int] = []
        current: int | None = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        frozen = tuple(path)
        self._shortest_path_cache[(start, goal)] = frozen
        self._shortest_path_cache[(goal, start)] = tuple(reversed(frozen))
        return path

    def clear_shortest_path_cache(self) -> None:
        """Clear process-local path state before a cold deterministic trial."""

        self._shortest_path_cache.clear()

    @property
    def shortest_path_cache_size(self) -> int:
        return len(self._shortest_path_cache)
