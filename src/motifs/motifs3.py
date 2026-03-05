"""
Efficient ad hoc motif counting algorithm for order 3 and 4 in undirected graphs. Unlike ESU, this algorithm logic is not based on enumeration but on combinatorial counting. Hopefully can be extended to weighted graphs
"""

from math import sqrt

from src.loaders import *
from src.stats import MotifStat


def count_motifs(hg: Hypergraph, order: int) -> dict[RawFrozenHypergraphUnWeighted, MotifStat]:
    assert order in (3, 4), "Only order 3 and 4 motifs are supported"
    assert not hg.has_multiedge()

    adj = hg.get_digraph_adj_list()
    adj_set = [set(neighbors) for neighbors in adj]
    adj_mat = hg.get_digraph_adj_matrix()

    distinct_motifs_count = 6 if order == 3 else 6
    stats = [MotifStat() for _ in range(distinct_motifs_count)]

    def count_3(vertex: int):
        nonlocal stats

        neighbors_count = len(adj[vertex])

        sum = 0
        for u in adj[vertex]:
            sum += sqrt(adj_mat[vertex][u])

        for i in range(len(adj[vertex]) - 1):
            u = adj[vertex][i]
            curr = sqrt(adj_mat[vertex][u])
            sum -= curr
            stats[0].intensity += curr * sum

        cross_edges = 0
        for u in adj[vertex]:
            for w in adj[u]:
                if w in adj_set[vertex] and w > u:
                    stats[0].intensity -= sqrt(adj_mat[vertex][u] * adj_mat[vertex][w])
                    stats[1].intensity += (adj_mat[vertex][u] * adj_mat[vertex][w] * adj_mat[u][w]) ** (1 / 3)
                    cross_edges += 1

        stats[0].count += (neighbors_count * (neighbors_count - 1) // 2) - cross_edges
        stats[1].count += cross_edges

    def count_4(vertex: int):
        nonlocal stats
        partial = [MotifStat() for _ in range(6)]
        for distal in adj[vertex]:
            common = adj_set[distal].intersection(adj_set[vertex])
            inner_nc = adj_set[vertex] - common - {distal}
            outer_nc = adj_set[distal] - common - {vertex}

            common_cross = 0
            for c in common:
                for u in adj[c]:
                    if c < u and u in common:
                        common_cross += 1

            inner_nc_cross = 0
            for nc in inner_nc:
                for u in adj[nc]:
                    if nc < u and u in inner_nc:
                        inner_nc_cross += 1

            inter_cross = 0
            for p in common:
                for u in adj[p]:
                    if u in adj_set[vertex] and u not in common and u != vertex and u != distal:
                        inter_cross += 1

            type5_cross = 0
            for o in outer_nc:
                for u in adj[o]:
                    if u in inner_nc:
                        type5_cross += 1

            partial[0].count += len(inner_nc) * (len(inner_nc) - 1) // 2 - inner_nc_cross
            partial[1].count += (len(adj[vertex]) - len(common) - 1) * len(common) - inter_cross
            partial[2].count += (len(common) * (len(common) - 1) // 2) - common_cross
            partial[3].count += common_cross
            partial[4].count += len(inner_nc) * len(outer_nc) - type5_cross
            partial[5].count += type5_cross

        stats[0].count += partial[0].count
        stats[1].count += partial[1].count
        stats[2].count += partial[2].count
        stats[3].count += partial[3].count
        stats[4].count += partial[4].count
        stats[5].count += partial[5].count

    # digraph counting
    for node in range(len(adj)):
        if order == 3:
            count_3(node)
        elif order == 4:
            count_4(node)

    # Normalize counts and intensities by the number of automorphisms of each
    normalizer_coefficient = {3: [1, 3, 1, 1, 1, 1], 4: [3, 2, 2, 12, 2, 8]}
    for i in range(distinct_motifs_count):
        stats[i].count //= normalizer_coefficient[order][i]
        stats[i].intensity /= normalizer_coefficient[order][i]

    # hypergraph counting
    if order == 3:
        for edge in hg.get_order_map().get(3, []):
            edge_weights = []
            e1 = hg.get_edges_by_nodes((edge.nodes[0], edge.nodes[1]))
            e2 = hg.get_edges_by_nodes((edge.nodes[0], edge.nodes[2]))
            e3 = hg.get_edges_by_nodes((edge.nodes[1], edge.nodes[2]))
            if len(e1) > 0:
                edge_weights.append(e1[0].weight_or(1.0))
            if len(e2) > 0:
                edge_weights.append(e2[0].weight_or(1.0))
            if len(e3) > 0:
                edge_weights.append(e3[0].weight_or(1.0))

            intensity_v = edge.weight_or(1.0)
            for w in edge_weights:
                intensity_v *= w
            intensity_v = intensity_v ** (1 / (len(edge_weights) + 1))

            # The motif discovered depends only on the number of contained diedges (+2 because the first 2 motifs are the ones without hyperedges)
            stats[2 + len(edge_weights)].count += 1
            stats[2 + len(edge_weights)].intensity += intensity_v

            inner_intensity_v = 1.0
            for w in edge_weights:
                inner_intensity_v *= w
            inner_intensity_v = inner_intensity_v ** (1 / len(edge_weights))
            if len(edge_weights) > 1:
                stats[len(edge_weights) - 2].count -= 1
                stats[len(edge_weights) - 2].intensity -= inner_intensity_v

    for i in range(distinct_motifs_count):
        stats[i].intensity /= stats[i].count if stats[i].count > 0 else 1

    rv = {}
    if order == 3:
        rv = {
            ((1, 2), (1, 3)): stats[0],
            ((1, 2), (1, 3), (2, 3)): stats[1],
            ((1, 2, 3),): stats[2],
            ((1, 2), (1, 2, 3)): stats[3],
            ((1, 2), (1, 2, 3), (1, 3)): stats[4],
            ((1, 2), (1, 2, 3), (1, 3), (2, 3)): stats[5],
        }

    return rv
