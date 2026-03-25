from __future__ import annotations

from dataclasses import dataclass

from src.graph import Hypergraph

from . import utils

MotifsRv = list[tuple[tuple[tuple[int, ...]], list[tuple[int, ...]]]]
Motif = tuple[tuple[int]]
MotifMapping = dict[Motif, set[Motif]]
MotifLabeling = dict[Motif, list[Motif]]


@dataclass
class MotifStat:
    count: int = 0
    intensity: float = 0
    coherence: float = 0
    actual_intensity: float = 0

    def __str__(self):
        return (
            f"MotifStat("
            f"count={self.count}, "
            f"intensity={self.intensity}, "
            f"mean_coherence={self.coherence}, "
            f"actual_intensity={self.actual_intensity}"
            f")"
        )

    __repr__ = __str__

    def __add__(self, other) -> MotifStat:
        count = self.count + other.count
        intensity = 0.0 if count == 0 else self.intensity * self.count / count + other.intensity * other.count / count
        coherence = 0.0 if count == 0 else self.coherence * self.count / count + other.coherence * other.count / count
        actual_intensity = (
            0.0
            if count == 0
            else self.actual_intensity * self.count / count + other.actual_intensity * other.count / count
        )

        return MotifStat(count, intensity, coherence, actual_intensity)

    def __iadd__(self, other) -> MotifStat:
        count = self.count + other.count
        intensity = 0.0 if count == 0 else self.intensity * self.count / count + other.intensity * other.count / count
        coherence = 0.0 if count == 0 else self.coherence * self.count / count + other.coherence * other.count / count
        actual_intensity = (
            0.0
            if count == 0
            else self.actual_intensity * self.count / count + other.actual_intensity * other.count / count
        )

        self.count = count
        self.intensity = intensity
        self.coherence = coherence
        self.actual_intensity = actual_intensity
        return self

    def __sub__(self, other) -> MotifStat:
        count = self.count - other.count
        intensity = 0.0 if count == 0 else self.intensity * self.count / count - other.intensity * other.count / count
        coherence = 0.0 if count == 0 else self.coherence * self.count / count - other.coherence * other.count / count
        actual_intensity = (
            0.0
            if count == 0
            else self.actual_intensity * self.count / count - other.actual_intensity * other.count / count
        )

        return MotifStat(count, intensity, coherence, actual_intensity)

    def __isub__(self, other) -> MotifStat:
        count = self.count - other.count
        intensity = 0.0 if count == 0 else self.intensity * self.count / count - other.intensity * other.count / count
        coherence = 0.0 if count == 0 else self.coherence * self.count / count - other.coherence * other.count / count
        actual_intensity = (
            0.0
            if count == 0
            else self.actual_intensity * self.count / count - other.actual_intensity * other.count / count
        )

        self.count = count
        self.intensity = intensity
        self.coherence = coherence
        self.actual_intensity = actual_intensity
        return self


# def aggregate(hg: Hypergraph, motifs: MotifsRv):
#     aggregated = []
#     hg.compute_adjacency()
#
#     for motif, instances in motifs:
#         count = len(motif)
#         total_intensity = 0
#         total_coherence = 0
#         total_actual_intensity = 0
#
#         for motif_instance in instances:
#             induced_subgraph = hg.get_induced_subgraph(motif_instance)
#             intensity = utils.intensity(induced_subgraph)
#             coherence = utils.coherence(induced_subgraph)
#
#             total_intensity += intensity
#             total_coherence += coherence
#             total_actual_intensity += intensity * coherence
#
#         aggregated.append(
#             MotifStat(
#                 count,
#                 total_intensity,
#                 total_coherence / count if count != 0 else 1,
#                 total_actual_intensity,
#             )
#         )
#     return aggregated
