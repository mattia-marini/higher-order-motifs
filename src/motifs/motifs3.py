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

    distinct_motifs_count = 2 if order == 3 else 6
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

    for node in range(len(adj)):
        if order == 4:
            count_4(node)
        elif order == 3:
            count_3(node)

    # Normalize counts and intensities by the number of automorphisms of each motif
    normalizer_coefficient = {3: [1, 3], 4: [3, 2, 2, 12, 2, 8]}
    for i in range(distinct_motifs_count):
        stats[i].count //= normalizer_coefficient[order][i]
        stats[i].intensity /= normalizer_coefficient[order][i] * stats[i].count if stats[i].count > 0 else 1

    rv = {}
    if order == 3:
        rv = {
            ((1, 2), (1, 3)): stats[0],
            ((1, 2), (1, 3), (2, 3)): stats[1],
        }

    return rv
