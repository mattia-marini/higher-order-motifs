"""
This file contains a series of classes representing hypergraphs and edges
"""
from typing import Tuple, Optional, Iterable, Set, Union
from enum import Enum
from dataclasses import dataclass


class Hyperedge:
    def __init__(self, nodes: Iterable[int], weight: Optional[float] = None):
        self.nodes: Tuple[int, ...] = tuple(nodes)
        self.weight: Optional[float] = weight

class Hypergraph:
    def __init__(self, edges: Iterable[Hyperedge]):
        self.edges: Set[Hyperedge] = set(edges)




class NormalizationMethod(Enum): 
    NONE = 1
    DEFAULT = 2
    BY_ORDER = 3
    RANKING = 4

@dataclass
class StandardConstructionMethod:
    """Standard construction method."""
    weighted: bool = False
    normalization_method: NormalizationMethod = NormalizationMethod.NONE

@dataclass
class TimeWindowConstructionMethod:
    """Time window-based construction method."""
    time_window: float = 1.0
    weighted: bool = False
    normalization_method: NormalizationMethod = NormalizationMethod.NONE

@dataclass
class TemporalPathConstructionMethod:
    """Placeholder for temporal path-based method."""
    # Add fields as needed in the future
    pass  # TODO: implement

class ConstructionMethod:
    """Factory for construction method instances."""

    @staticmethod
    def standard(
        weighted: bool = False, 
        normalization_method: NormalizationMethod = NormalizationMethod.NONE
    ) -> StandardConstructionMethod:
        return StandardConstructionMethod(weighted, normalization_method)

    @staticmethod
    def time_window(
        time_window: float, 
        weighted: bool = False, 
        normalization_method: NormalizationMethod = NormalizationMethod.NONE
    ) -> TimeWindowConstructionMethod:
        return TimeWindowConstructionMethod(time_window, weighted, normalization_method)

    @staticmethod
    def temporal_path() -> TemporalPathConstructionMethod:
        return TemporalPathConstructionMethod()

# Example usage
ConstructionMethod.standard(False, NormalizationMethod.DEFAULT)
ConstructionMethod.time_window(10.0, True, NormalizationMethod.RANKING)
