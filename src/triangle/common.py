from collections import deque
from typing import Any, Callable, Iterable

from src.graph import Hyperedge, Hypergraph
from src.motifs.motifs3 import count_motifs as count_motifs3
from tests.util import Colors, Loader, StandardConstructionMethod, time_function


def bfs(adj: list[list[int]], start_vertex=0):
    if len(adj) == 0:
        return []
    queue = deque([start_vertex])
    visited = [False] * len(adj)
    levels = [-1] * len(adj)
    visited[start_vertex] = True
    levels[start_vertex] = 0

    while queue:
        n = queue.popleft()
        for u in adj[n]:
            if not visited[u]:
                visited[u] = True
                levels[u] = levels[n] + 1
                queue.append(u)

    return levels


def counting_sort(v: list[Any], key: Callable[[Any], int] = lambda x: x, max_value=None):
    if max_value is None:
        max_value = max(v) if len(v) > 0 else 0
    buckets = [[] for _ in range(max_value + 1)]
    for e in v:
        buckets[key(e)].append(e)

    curr_idx = 0
    for i in range(len(buckets)):
        if buckets[i]:
            for e in buckets[i]:
                v[curr_idx] = e
                curr_idx += 1


def common_neighbors_sorted_list(u: list[int], v: list[int]) -> list[int]:
    common = []
    pointer_u = 0
    pointer_v = 0
    while pointer_u < len(u) and pointer_v < len(v):
        if u[pointer_u] == v[pointer_v]:
            common.append(u[pointer_u])
            pointer_u += 1
            pointer_v += 1
        elif u[pointer_u] < v[pointer_v]:
            pointer_u += 1
        else:
            pointer_v += 1

    return common
