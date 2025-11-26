import math
import numbers
from typing import Iterable


def diff_sum(original, null_models):
    u_null = avg(null_models)

    res = []
    for i in range(len(original)):
        res.append((original[i][1] - u_null[i]) / (original[i][1] + u_null[i] + 4))

    return res


def norm_vector(a):
    res = []
    M = 0
    for i in a:
        M += i**2
    M = math.sqrt(M)
    res = [i / M for i in a]
    return res


def count(edges):
    """
    Prints information about the hypergraph:
    - number of nodes
    - number of hyperedges
    - count of hyperedges of sizes 2 to 5.
    """
    d = {}
    n = set()
    for i in range(2, 6):
        d[i] = 0
    for i in edges:
        try:
            d[len(i)] += 1
        except:
            pass
        finally:
            for j in i:
                n.add(j)
    print(
        "& {} & {} & {} & {} & {} & {}".format(
            len(n), len(edges), d[2], d[3], d[4], d[5]
        )
    )


def count_weight(edges):
    """
    Prints information about the hypergraph:
    - number of nodes
    - number of hyperedges
    - count of hyperedges of sizes 2 to 5 and the average weight
    """
    d = {}
    w = {}
    n = set()
    for i in range(2, 6):
        d[i] = 0
        w[i] = 0
    for edge, weight in edges.items():
        try:
            d[len(edge)] += 1
            w[len(edge)] += weight
        except:
            pass
        finally:
            for j in edge:
                n.add(j)

    for i in range(2, 6):
        if d[i] == 0:
            w[i] = 0
        else:
            w[i] = w[i] / d[i]
    print(
        "& {} & {} & {}:{:.3f} & {}:{:.3f} & {}:{:.3f} & {}:{:.3f}".format(
            len(n), len(edges), d[2], w[2], d[3], w[3], d[4], w[4], d[5], w[5]
        )
    )


def avg(motifs):
    result = []
    for i in range(len(motifs[0])):
        s = 0
        for j in range(len(motifs)):
            s += motifs[j][i][1]

        result.append(s / len(motifs))
    return result


def sigma(motifs):
    u = avg(motifs)

    result = []
    for i in range(len(motifs[0])):
        s = 0
        for j in range(len(motifs)):
            s += (motifs[j][i][1] - u[i]) ** 2
        s /= len(motifs)
        s = s**0.5

        result.append(s)
    return result


def z_score(original, null_models):
    u_null = avg(null_models)
    sigma_null = sigma(null_models)

    z_scores = []
    for i in range(len(original)):
        z_scores.append((original[i][1] - u_null[i]) / (sigma_null[i] + 0.01))

    return z_scores


def power_set(A: Iterable[int]):
    """
    Generate the power set of a given set A.

    Args:
        A (list): The input set.

    Returns:
        list: A list containing all subsets of A.
    """

    A = [x for x in A]
    A = sorted(A)

    subsets = []
    N = len(A)

    for mask in range(1 << N):
        subset = []

        for n in range(N):
            if ((mask >> n) & 1) == 1:
                subset.append(A[n])

        subsets.append(subset)

    return subsets


def is_connected(edges, N):
    nodes = set()
    for e in edges:
        for n in e:
            nodes.add(n)

    if len(nodes) != N:
        return False

    visited = {}
    for i in nodes:
        visited[i] = False
    graph = {}
    for i in nodes:
        graph[i] = []

    for edge in edges:
        for i in range(len(edge)):
            for j in range(len(edge)):
                if edge[i] != edge[j]:
                    graph[edge[i]].append(edge[j])
                    graph[edge[j]].append(edge[i])

    q = []
    nodes = list(nodes)
    q.append(nodes[0])
    while len(q) != 0:
        v = q.pop(len(q) - 1)
        if not visited[v]:
            visited[v] = True
            for i in graph[v]:
                q.append(i)
    conn = True
    for i in nodes:
        if not visited[i]:
            conn = False
            break
    return conn


def relabel(edges, relabeling):
    """
    Relabel the vertices of the edges according to the given relabeling.

    Args:
        edges (list[tuple[int]]): The list of edges to be relabeled.
        relabeling (tuple[int]): A tuple representing the new labels for the vertices.

    Returns:
        list[tuple[int]]: The relabeled edges, sorted.
    """
    res = []
    for edge in edges:
        new_edge = []
        for v in edge:
            new_edge.append(relabeling[v - 1])
        res.append(tuple(sorted(new_edge)))
    return sorted(res)


def intensity(edges):
    # print(edges)
    if isinstance(edges, set):
        return 1.0

    weights = [w for _, w in edges.items()]

    log_sum = sum(math.log(n) for n in weights)
    return math.exp(log_sum / len(weights))


def coherence(edges):
    if isinstance(edges, set):
        return 1.0

    mean = sum(edges.values()) / len(edges)
    return intensity(edges) / mean


def induced_subgraph(edges, nodes):
    subgraph = {}
    node_set = set(nodes)
    for e, w in edges.items():
        include = True
        for n in e:
            if n not in node_set:
                include = False
                break
        if include:
            subgraph[e] = w
    return subgraph


def assert_hypergraph(edges, weighted):
    """
    Checks if the given edges represent a weighted/unweighted hypergraph.
    """
    if weighted:
        if not isinstance(edges, dict):
            raise TypeError(
                "Weighted hypergraph should be represented as a dict[tuple[int],numbers.Number]"
            )

        for edge, weight in edges.items():
            if not (isinstance(edge, tuple) and all(isinstance(k, int) for k in edge)):
                raise TypeError("Each edge must be a tuple of integers")
            if not isinstance(weight, numbers.Number):
                raise TypeError("Weights must be numeric values")
    else:
        if not isinstance(edges, set):
            raise TypeError(
                "Unweighted hypergraph should be represented as a set[tuple[int]]"
            )
        for edge in edges:
            if not (isinstance(edge, tuple) and all(isinstance(k, int) for k in edge)):
                raise TypeError("Each edge must be a tuple of integers")


def normalize_weights(edges):
    max_val = max(edges.values())
    for k in edges:
        edges[k] = edges[k] / max_val


# out = len(isom_classes.keys())
# print(out)
