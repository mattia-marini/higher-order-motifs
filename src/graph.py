"""
This file contains a series of classes representing hypergraphs and edges
"""
from typing import Tuple, Optional, Iterable, Set, Union, Any, Callable
from enum import Enum
from dataclasses import dataclass

class Hyperedge: 
    def __init__(self, nodes: Iterable[int], weight: Optional[float] = None, payload: Optional[Any] = None):
        self._id : Optional[int] = None
        self.nodes: Tuple[int, ...] = tuple(sorted(nodes))
        self._weight: Optional[float] = weight
        self.payload: Optional[Any] = payload

    def __hash__(self):
        return hash(self.nodes)

    def __eq__(self, other):
        return isinstance(other, Hyperedge) and self.nodes == other.nodes

    def weight(self) -> float:
        if self._weight is not None:
            return self._weight
        else:
            raise ValueError("Called weight on unweighted edge")

    def weight_or(self, default: float = 1.0) -> float:
        if self._weight is not None:
            return self._weight
        else:
            return default

    def weight_or_else(self, default: Callable[['Hyperedge'], float]) -> float:
        if self._weight is not None:
            return self._weight
        else:
            return default(self)

class Hypergraph:
    def __init__(self):
        self.edges: Set[Hyperedge] = set()
        self.next_edge_id: int = 0

    def add_edge(self, edge: Hyperedge):
        self.next_edge_id += 1
        edge._id = self.next_edge_id
        self.edges.add(edge)
        return self.next_edge_id
