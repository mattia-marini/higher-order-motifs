from dataclasses import dataclass

from src.graph import Hypergraph
from src.motifs.motifs_base import MotifsRv

from . import utils


@dataclass()
class AggregatedInfos:
    count: int
    intensity: float
    mean_coherence: float
    actual_intensity: float

    def __str__(self):
        return (
            f"AggregatedInfos("
            f"count={self.count}, "
            f"intensity={self.intensity}, "
            f"mean_coherence={self.mean_coherence}, "
            f"actual_intensity={self.actual_intensity}"
            f")"
        )

    __repr__ = __str__


def aggregate(hg: Hypergraph, motifs: MotifsRv):
    aggregated = []
    hg.compute_adjacency()

    for motif, instances in motifs:
        count = len(motif)
        total_intensity = 0
        total_coherence = 0
        total_actual_intensity = 0

        for motif_instance in instances:
            induced_subgraph = hg.get_induced_subgraph(motif_instance)
            intensity = utils.intensity(induced_subgraph)
            coherence = utils.coherence(induced_subgraph)

            total_intensity += intensity
            total_coherence += coherence
            total_actual_intensity += intensity * coherence

        aggregated.append(
            AggregatedInfos(
                count,
                total_intensity,
                total_coherence / count if count != 0 else 1,
                total_actual_intensity,
            )
        )
    return aggregated
