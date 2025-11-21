from . import utils

class AggregatedInfos:
    def __init__(self, count, total_intensity, mean_coherence, actual_intensity):
        self.count  = count
        self.intensity = total_intensity
        self.mean_coherence = mean_coherence
        self.actual_intensity = actual_intensity
    def __str__(self):
        return (f"AggregatedInfos("
                f"count={self.count}, "
                f"intensity={self.intensity}, "
                f"mean_coherence={self.mean_coherence}, "
                f"actual_intensity={self.actual_intensity}"
                f")")
    __repr__ = __str__ 

def aggregate(edges, motifs):
    adj = {}
    for e in edges:
        for node in e:
            if node not in adj:
                adj[node] = []

            adj[node].append(e)

    def induced_subgraph(nodes):
        if len(nodes) == 0:
            return []
        rv = set()
        nodes = set(nodes)

        for n in nodes:
            for e in adj.get(n, []):
                if all(x in nodes for x in e):
                    rv.add(e)
        return rv



    aggregated = []
    for motif in motifs:
        count = len(motif)
        total_intensity = 0
        total_coherence = 0
        total_actual_intensity = 0

        for motif_instance in motif:
            induced_subgraph_t = induced_subgraph(motif_instance)
            intensity = utils.intensity(induced_subgraph_t)
            coherence = utils.coherence(induced_subgraph_t)
            total_intensity += intensity
            total_coherence += coherence
            total_actual_intensity += intensity * coherence

        aggregated.append(AggregatedInfos(count, total_intensity, total_coherence / count if count != 0 else 1, total_actual_intensity))
    return aggregated
