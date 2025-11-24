"""
This file contains a series of classes representing hypergraphs and edges
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional, Iterable, Set, Union, Any, Callable, List
from enum import Enum
from dataclasses import dataclass

class Hyperedge: 
    def __init__(self, nodes: Iterable[int], weight: Optional[float] = None, payload: Optional[Any] = None):
        self.nodes: Tuple[int, ...] = tuple(sorted(nodes))
        self._weight: Optional[float] = weight
        self.payload: Optional[Any] = payload

    # def __hash__(self):
    #     return hash(self.nodes)
    # def __eq__(self, other):
    #     return isinstance(other, Hyperedge) and self.nodes == other.nodes

    def __str__(self) -> str:
        if self._weight is None:
            return f"({self.nodes})"
        else:
            return f"({self.nodes}, {self._weight})"
    __repr__ = __str__

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

class HyperedgeHandle: 
    """
    This class serves as a wrapper around an Hyperedge in the moment it is
    inserted into a graph. This allows for safe modification
    """
    def __init__(self, graph: Hypergraph, edge: Hyperedge, id : int):
        self._graph = graph
        self._id = id
        self._edge = edge

    def __str__(self) -> str:
        return f"HyperedgeHandle(id={self._id}, edge={self._edge})"

    __repr__ = __str__

    @property
    def nodes(self) -> Tuple[int, ...]:
        return self._edge.nodes

    @nodes.setter
    def nodes(self, nodes: Iterable[int]):
        nodes = tuple(sorted(nodes))
        if self._id in self._graph._edges:
            self._graph._nodes_map[self._edge.nodes].discard(self._id)
            self._graph._nodes_map[nodes].add(self._id)

        self._edge.nodes = nodes
    
    def weight(self) -> float:
        return self.weight()

    def weight_or(self, default: float = 1.0) -> float:
        return self.weight_or(default)

    def weight_or_else(self, default: Callable[[Hyperedge], float]) -> float:
        return self.weight_or_else(default)

class Hypergraph:
    def __init__(self):
        self._edges: Dict[int, Hyperedge] = {}
        self._nodes_map: Dict[tuple[int,...], Set[int]] = {} # For constant time access from nodes
        self._handles: Dict[int, HyperedgeHandle] = {}
        self._next_edge_id: int = 0

    def __str__(self) -> str:
        return f"Hypergraph(num_edges={len(self._edges)})"


    def add_edge(self, edge: Hyperedge):
        id = self._next_edge_id
        if edge.nodes not in self._nodes_map:
            self._nodes_map[edge.nodes] = set()
        self._edges[id] = edge
        self._nodes_map[edge.nodes].add(id)
        self._handles[id] = HyperedgeHandle(self, edge, id)
        self._next_edge_id += 1
        return id

    def remove_edge(self, id: int) -> Hyperedge:
        if id not in self._edges:
            raise KeyError(f"Edge id {id} not found in graph")
        self._nodes_map[self._edges[id].nodes].remove(id)
        self._handles.pop(id)
        return self._edges.pop(id)

    def get_edge_by_id(self, edge_id: int) -> HyperedgeHandle:
        if edge_id not in self._edges:
            raise KeyError(f"Edge id {id} not found in graph")
        return self._handles[edge_id]

    def get_edges_by_nodes(self, nodes: Iterable[int]) -> List[HyperedgeHandle]:
        nodes = tuple(sorted(nodes))
        if nodes not in self._nodes_map:
            return []
        rv = []
        for edge_id in self._nodes_map[nodes]:
            rv.append(self._handles[edge_id])
        return rv

    def get_first_edges_by_nodes(self, nodes: Iterable[int]) -> HyperedgeHandle:
        rv = self.get_edges_by_nodes(nodes)
        if len(rv) == 0: 
            raise KeyError(f"No edge with nodes {nodes}")
        return rv[0]
