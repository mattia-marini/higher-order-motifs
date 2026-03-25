"""
Efficient ad hoc motif counting algorithm for order 3 and 4 in undirected graphs. Unlike ESU, this algorithm logic is not based on enumeration but on combinatorial counting. Hopefully can be extended to weighted graphs
"""

from math import prod, sqrt

from src.loaders import *
from src.motifs.motifs_count_base import generate_motifs
from src.stats import MotifStat
from src.utils import relabel_unweighted


def count_3(hg: Hypergraph) -> dict[RawFrozenHypergraphUnWeighted, MotifStat]:
    adj = hg.get_digraph_adj_list()
    adj_set = [set(neighbors) for neighbors in adj]
    adj_mat = hg.get_digraph_adj_matrix()

    distinct_motifs_count = 6
    stats = [MotifStat() for _ in range(distinct_motifs_count)]

    def count_diedge_motifs(vertex: int):
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

    # digraph counting
    for node in range(len(adj)):
        count_diedge_motifs(node)

    # Normalize counts and intensities by the number of automorphisms of each
    normalizer_coefficient = [1, 3, 1, 1, 1, 1]
    for i in range(distinct_motifs_count):
        stats[i].count //= normalizer_coefficient[i]
        stats[i].intensity /= normalizer_coefficient[i]

    # hypergraph counting
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

    rv = {
        ((1, 2), (1, 3)): stats[0],
        ((1, 2), (1, 3), (2, 3)): stats[1],
        ((1, 2, 3),): stats[2],
        ((1, 2), (1, 2, 3)): stats[3],
        ((1, 2), (1, 2, 3), (1, 3)): stats[4],
        ((1, 2), (1, 2, 3), (1, 3), (2, 3)): stats[5],
    }

    return rv


def count_4(hg: Hypergraph) -> dict[RawFrozenHypergraphUnWeighted, MotifStat]:
    adj = hg.get_digraph_adj_list()
    adj_set = [set(neighbors) for neighbors in adj]
    adj_mat = hg.get_digraph_adj_matrix()

    distinct_motifs_count = 171
    # stats = [MotifStat() for _ in range(distinct_motifs_count)]

    # digraph motifs aliases
    type0 = ((1, 2), (1, 3), (1, 4))
    type1 = ((1, 2), (1, 3), (1, 4), (2, 3))
    type2 = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4))
    type3 = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    type4 = ((1, 2), (1, 3), (2, 4))
    type5 = ((1, 2), (1, 3), (2, 4), (3, 4))

    rep, rep_map = generate_motifs(4)
    result = {rep: MotifStat() for rep in rep}

    test = 0

    def count_diedge_motifs(vertex: int):
        nonlocal result, test
        partial = [MotifStat() for _ in range(6)]
        for distal in adj[vertex]:
            common = adj_set[distal].intersection(adj_set[vertex])
            inner_nc = adj_set[vertex] - common - {distal}
            outer_nc = adj_set[distal] - common - {vertex}

            # intensity
            pivot_edge_cr = adj_mat[vertex][distal] ** (1 / 3)
            type0_sum = 0
            inner_nc_list = list(inner_nc)
            for i in range(len(inner_nc_list)):
                type0_sum += adj_mat[vertex][inner_nc_list[i]] ** (1 / 3)

            for i in range(len(inner_nc_list) - 1):
                curr_weight_cr = adj_mat[vertex][inner_nc_list[i]] ** (1 / 3)
                type0_sum -= curr_weight_cr
                partial[0].intensity += pivot_edge_cr * curr_weight_cr * type0_sum

            inner_sum = sum(adj_mat[vertex][inc] ** (1 / 3) for inc in inner_nc)
            outer_sum = sum(adj_mat[distal][onc] ** (1 / 3) for onc in outer_nc)
            partial[4].intensity += pivot_edge_cr * inner_sum * outer_sum

            inner_nc_cross = 0
            for nc1 in inner_nc:
                for nc2 in adj[nc1]:
                    if nc1 < nc2 and nc2 in inner_nc:
                        inner_nc_cross += 1
                        curr = adj_mat[vertex][nc1] * adj_mat[vertex][nc2] * adj_mat[vertex][distal]
                        partial[0].intensity -= curr ** (1 / 3)
                        partial[1].intensity += (curr * adj_mat[nc1][nc2]) ** (1 / 4)

            inter_cross = 0
            for c in common:
                for nc in adj[c]:
                    if nc in inner_nc:
                        inter_cross += 1
                        curr = adj_mat[vertex][c] * adj_mat[vertex][nc] * adj_mat[vertex][distal] * adj_mat[c][distal]
                        # partial[1].intensity -= curr ** (1 / 4)
                        partial[2].intensity += (curr * adj_mat[c][nc]) ** (1 / 5)

            common_cross = 0
            for c1 in common:
                for c2 in adj[c1]:
                    if c1 < c2 and c2 in common:
                        common_cross += 1
                        curr = (
                            adj_mat[vertex][c1]
                            * adj_mat[vertex][c2]
                            * adj_mat[c1][distal]
                            * adj_mat[c2][distal]
                            * adj_mat[vertex][distal]
                        )
                        # partial[1].intensity -= curr ** (1 / 4)
                        partial[3].intensity += (curr * adj_mat[c1][c2]) ** (1 / 6)

            type5_cross = 0
            for onc in outer_nc:
                for inc in adj[onc]:
                    if inc in inner_nc:
                        type5_cross += 1
                        curr = adj_mat[vertex][distal] * adj_mat[vertex][inc] * adj_mat[distal][onc]
                        partial[4].intensity -= curr ** (1 / 3)
                        partial[5].intensity += (curr * adj_mat[inc][onc]) ** (1 / 4)

            partial[0].count += len(inner_nc) * (len(inner_nc) - 1) // 2 - inner_nc_cross
            partial[1].count += inner_nc_cross
            partial[2].count += inter_cross
            partial[3].count += common_cross
            partial[4].count += len(inner_nc) * len(outer_nc) - type5_cross
            partial[5].count += type5_cross

        result[type0].count += partial[0].count
        result[type0].intensity += partial[0].intensity

        result[type1].count += partial[1].count
        result[type1].intensity += partial[1].intensity

        result[type2].count += partial[2].count
        result[type2].intensity += partial[2].intensity

        result[type3].count += partial[3].count
        result[type3].intensity += partial[3].intensity

        result[type4].count += partial[4].count
        result[type4].intensity += partial[4].intensity

        result[type5].count += partial[5].count
        result[type5].intensity += partial[5].intensity

    # digraph counting
    for node in range(len(adj)):
        count_diedge_motifs(node)

    # Normalize counts and intensities by the number of automorphisms of each
    normalizer_coefficient = [3, 1, 4, 12, 2, 8]

    result[type0].intensity /= max(1, result[type0].count)
    result[type1].intensity /= max(1, result[type1].count)
    result[type2].intensity /= max(1, result[type2].count)
    result[type3].intensity /= max(1, result[type3].count)
    result[type4].intensity /= max(1, result[type4].count)
    result[type5].intensity /= max(1, result[type5].count)

    result[type0].count //= normalizer_coefficient[0]
    result[type1].count //= normalizer_coefficient[1]
    result[type2].count //= normalizer_coefficient[2]
    result[type3].count //= normalizer_coefficient[3]
    result[type4].count //= normalizer_coefficient[4]
    result[type5].count //= normalizer_coefficient[5]

    # order 3 hypergraph
    hg_adj = hg.get_adjacency_immut_ref()
    for edge in hg.get_order_map().get(3, []):
        ext_set = set()
        for node in edge.nodes:
            ext_set.update(hg_adj[node])
        ext_set = [e for e in ext_set if e.order < 4]

        # print(ext_set)
        ext_edges = {}
        for u in ext_set:
            out_count = 0
            out_node = None
            for node in u.nodes:
                if node not in edge.nodes:
                    out_count += 1
                    out_node = node
            if out_count == 1:
                if out_node not in ext_edges:
                    ext_edges[out_node] = []
                ext_edges[out_node].append(u)

        induced_subgraph = hg.get_induced_subgraph(edge.nodes)
        induced_subgraph = [e for e in induced_subgraph if len(e.nodes) <= 3]

        center_unweighted = [edge.to_raw_hyperedge_unweighted() for edge in induced_subgraph]

        center_prod = prod([edge.weight for edge in induced_subgraph])
        center_diedge_prod = prod([edge.weight for edge in induced_subgraph if len(edge.nodes) == 2])

        for u, edges in ext_edges.items():
            min_id = min([edge.id] + [e.id for e in edges if len(e.nodes) == 3])
            if edge.id != min_id:
                continue

            new_motif = center_unweighted + [e.to_raw_hyperedge_unweighted() for e in edges]

            tot_prod = center_prod * prod([edge.weight for edge in edges])
            tot_intensity = tot_prod ** (1 / (len(edges) + len(induced_subgraph)))
            stat = MotifStat(count=1, intensity=tot_intensity)

            mapping = {k: v for k, v in zip(edge.nodes + (u,), range(1, 5))}
            labeled_new_motif = relabel_unweighted(new_motif, mapping)

            result[rep_map[labeled_new_motif]] += stat

            underlying_diedges = tuple([e for e in labeled_new_motif if len(e) == 2])
            if underlying_diedges in rep_map:  # if its connected
                # print("Found motif with underlying diedges: ", underlying_diedges)
                diedges_prod = center_diedge_prod * prod([e.weight for e in edges if len(e.nodes) == 2])
                tot_diedge_intensity = diedges_prod ** (1 / len(underlying_diedges))

                result[rep_map[underlying_diedges]] -= MotifStat(count=1, intensity=tot_diedge_intensity)

    # return result
    for edge in hg.get_order_map().get(4, []):
        induced_subgraph = hg.get_induced_subgraph(edge.nodes)
        intensity_v = prod([e.weight for e in induced_subgraph]) ** (1 / len(induced_subgraph))
        new_motif = [e.to_raw_hyperedge_unweighted() for e in induced_subgraph]

        mapping = {k: v for k, v in zip(edge.nodes, range(1, 5))}
        labeled_new_motif = relabel_unweighted(new_motif, mapping)

        result[rep_map[labeled_new_motif]] += MotifStat(count=1, intensity=intensity_v)

        underlying_motif = tuple([e for e in labeled_new_motif if len(e) == 2 or len(e) == 3])
        if underlying_motif in rep_map:  # if its connected
            underlying_intensity = (intensity_v ** len(induced_subgraph) / edge.weight) ** (1 / len(underlying_motif))
            result[rep_map[underlying_motif]] -= MotifStat(count=1, intensity=underlying_intensity)

    return result


def count_motifs(hg: Hypergraph, order: int) -> dict[RawFrozenHypergraphUnWeighted, MotifStat]:
    assert order in (3, 4), "Only order 3 and 4 motifs are supported"
    assert not hg.has_multiedge()

    if order == 3:
        return count_3(hg)
    else:
        return count_4(hg)
