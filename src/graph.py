"""
This file contains a series of classes representing hypergraphs and edges
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, cast

from src.utils import power_set


class NormalizationMethod(Enum):
    NONE = 1
    DEFAULT = 2
    ORDER = 3
    RANKING = 4


@dataclass
class ConstructionMethodBase:
    limit_edge_size: Optional[int] = None
    normalization_method: NormalizationMethod = NormalizationMethod.NONE
    weighted: bool = False

    def description(self) -> str:
        return f"{self.limit_edge_size}{'W' if self.weighted else 'w'}n{self.normalization_method.name.lower()}"


@dataclass
class StandardConstructionMethod(ConstructionMethodBase):
    pass


@dataclass
class TimeWindowConstructionMethod(ConstructionMethodBase):
    time_window: float = 1.0

    def description(self) -> str:
        super_desc = super().description()
        return f"{super_desc}tw{self.time_window:.2f}"


@dataclass
class TemporalPathConstructionMethod(ConstructionMethodBase):
    pass


class Hyperedge:
    def __init__(
        self,
        nodes: Iterable[int],
        weight: Optional[float] = None,
        payload: Optional[Any] = None,
    ):
        self.nodes: Tuple[int, ...] = tuple(sorted(nodes))
        self._weight: Optional[float] = weight
        self.payload: Optional[Any] = payload

    def __str__(self) -> str:
        if self._weight is None:
            return f"({self.nodes})"
        else:
            return f"({self.nodes}, {self._weight})"

    __repr__ = __str__

    def to_tuple(
        self,
    ) -> tuple[
        tuple[int, ...],
        Optional[float],
    ]:
        return (
            self.nodes,
            self._weight,
        )

    def weighted(self) -> bool:
        return self._weight is not None

    @property
    def order(self) -> int:
        return len(self.nodes)

    @order.setter
    def order(self, value: int):
        pass

    @property
    def weight(self) -> float:
        if self._weight is not None:
            return self._weight
        else:
            raise ValueError("Called weight on unweighted edge")

    @weight.setter
    def weight(self, value: float):
        self._weight = value

    def weight_or(self, default: float = 1.0) -> float:
        if self._weight is not None:
            return self._weight
        else:
            return default

    def weight_or_else(self, default: Callable[["Hyperedge"], float]) -> float:
        if self._weight is not None:
            return self._weight
        else:
            return default(self)


class HyperedgeHandle:
    """
    This class serves as a wrapper around an Hyperedge in the moment it is
    inserted into a graph. This allows for safe modification
    """

    def __init__(self, graph: Hypergraph, edge: Hyperedge, id: int):
        self._graph = graph
        self._id = id
        self._edge = edge

    def __str__(self) -> str:
        return f"HyperedgeHandle(id={self._id}, edge={self._edge})"

    __repr__ = __str__

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, HyperedgeHandle):
            return False
        return self._id == other._id

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

    def to_tuple(
        self,
    ) -> tuple[
        tuple[int, ...],
        Optional[float],
    ]:
        return self._edge.to_tuple()

    @property
    def order(self) -> int:
        return self._edge.order

    @order.setter
    def order(self, value: int):
        pass

    @property
    def weight(self) -> float:
        return self._edge.weight

    @weight.setter
    def weight(self, weight: float):
        self._edge.weight = weight

    def weight_or(self, default: float = 1.0) -> float:
        return self.weight_or(default)

    def weight_or_else(self, default: Callable[[Hyperedge], float]) -> float:
        return self.weight_or_else(default)

    def weighted(self) -> bool:
        return self._edge.weighted()


class Hypergraph:
    def __init__(self):
        self._edges: Dict[int, Hyperedge] = {}
        self._nodes_map: Dict[
            tuple[int, ...], Set[int]
        ] = {}  # For constant time access from nodes
        self._handles: Dict[int, HyperedgeHandle] = {}
        self._next_edge_id: int = 0
        self._weighted: Optional[bool] = None
        self._nodes: Set[int] = set()
        self._adjacency: Optional[Dict[int, List[HyperedgeHandle]]] = None

    def __str__(self) -> str:
        return f"Hypergraph(num_edges={len(self._edges)})"

    @property
    def n(self) -> int:
        return len(self._nodes)

    @n.setter
    def n(self, value: int):
        pass

    @property
    def e(self) -> int:
        return len(self._edges)

    @e.setter
    def e(self, value: int):
        pass

    @property
    def edges(self) -> Iterable[HyperedgeHandle]:
        return [handle for handle in self._handles.values()]

    @edges.setter
    def edges(self, value: Iterable[HyperedgeHandle]):
        pass

    def add_edge(self, edge: Hyperedge):
        if self._weighted is None:
            self._weighted = edge.weighted()

        if self._weighted != edge.weighted():
            raise ValueError("Cannot add edge with different weight type to graph")

        for n in edge.nodes:
            self._nodes.add(n)

        id = self._next_edge_id
        if edge.nodes not in self._nodes_map:
            self._nodes_map[edge.nodes] = set()
        self._edges[id] = edge
        self._nodes_map[edge.nodes].add(id)
        self._handles[id] = HyperedgeHandle(self, edge, id)
        self._next_edge_id += 1
        if self._adjacency is not None:
            for node in edge.nodes:
                if node not in self._adjacency:
                    self._adjacency[node] = []
                self._adjacency[node].append(self._handles[id])
        return id

    def remove_edge(self, id: int) -> Hyperedge:
        if id not in self._edges:
            raise KeyError(f"Edge id {id} not found in graph")

        edge_handle = self._handles.pop(id)
        if self._adjacency is not None:
            for n in edge_handle.nodes:
                self._adjacency[n].remove(edge_handle)
        self._nodes_map[self._edges[id].nodes].remove(id)
        edge = self._edges.pop(id)

        return edge

    def get_edge_by_id(self, edge_id: int) -> HyperedgeHandle:
        if edge_id not in self._edges:
            raise KeyError(f"Edge id {id} not found in graph")
        return self._handles[edge_id]

    def has_edge_with_id(self, edge_id: int) -> bool:
        return edge_id in self._edges

    def has_edge_with_nodes(self, nodes: Iterable[int]) -> bool:
        nodes = tuple(sorted(nodes))
        return nodes in self._nodes_map and len(self._nodes_map[nodes]) > 0

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

    def get_order_map(self) -> Dict[int, List[HyperedgeHandle]]:
        rv = {}
        for handle in self._handles.values():
            if handle.order not in rv:
                rv[handle.order] = []
            rv[handle.order].append(handle)
        return rv

    def normalize_weights(
        self, normalization_method: NormalizationMethod = NormalizationMethod.DEFAULT
    ):
        if self._weighted is False and normalization_method != NormalizationMethod.NONE:
            raise ValueError("Cannot normalize weights of unweighted graph")

        if normalization_method == NormalizationMethod.NONE:
            pass

        elif normalization_method == NormalizationMethod.DEFAULT:
            max_weight = max(edge.weight for edge in self._edges.values())
            for _, edge in self._edges.items():
                edge.weight = edge.weight / max_weight

        elif normalization_method == NormalizationMethod.ORDER:
            order_map = self.get_order_map()
            for order, nodes in order_map.items():
                max_weight = max(edge.weight for edge in nodes)
                for edge in nodes:
                    edge.weight = edge.weight / max_weight

        elif normalization_method == NormalizationMethod.RANKING:
            raise NotImplementedError("Ranking normalization not implemented yet")

    def hash(self):
        """
        Returns a SHA-1 hash of a dictionary.
        """

        data_array = [edge.to_tuple() for edge in self._edges.values()]
        data_array.sort()

        encoded = json.dumps(data_array).encode()
        return hashlib.sha1(encoded).hexdigest()

    def compute_adjacency(self):
        """
        This function computes the adjacency list of the hypergraph. Enables
        efficient executions for certain methods, such as induced subgraph
        computation
        """
        if self._adjacency is not None:
            return

        self._adjacency = {}
        for edge_handle in self._handles.values():
            for node in edge_handle.nodes:
                if node not in self._adjacency:
                    self._adjacency[node] = []
                self._adjacency[node].append(edge_handle)

    def get_adjacency_copy(self) -> Dict[int, List[HyperedgeHandle]]:
        self.compute_adjacency()
        if self._adjacency is None:
            raise ValueError("Adjacency not computed")
        return self._adjacency.copy()

    def get_adjacency_mut(self) -> Dict[int, List[HyperedgeHandle]]:
        """
        One should not modify the returned adjacency directly, as it would mess
        with the internal logic of the graph
        """
        self.compute_adjacency()
        if self._adjacency is None:
            raise ValueError("Adjacency not computed")
        return self._adjacency

    def get_induced_subgraph(self, nodes: Iterable[int]) -> Set[HyperedgeHandle]:
        nodes = set(nodes)
        rv = set()

        def power_set_method():
            p_nodes = power_set(nodes)
            for edge in p_nodes:
                if len(edge) >= 2:
                    rv.update(self.get_edges_by_nodes(edge))

        def adjacency_method():
            if self._adjacency is None:
                raise ValueError("Adjacency not computed")
            for n in nodes:
                if n not in self._adjacency:
                    raise ValueError(
                        f"Cannot get induced subgraph. Node {n} is not in graph"
                    )

                for edge_handle in self._adjacency[n]:
                    contained = True
                    for en in edge_handle.nodes:
                        if en not in nodes:
                            contained = False
                            break
                    if contained:
                        rv.add(edge_handle)

        if self._adjacency is None:
            power_set_method()
        else:
            adjacency_method()

        return rv

    def has_multiedge(self) -> bool:
        for edge_ids in self._nodes_map.values():
            if len(edge_ids) > 1:
                return True
        return False
