from collections import deque
from typing import Iterable

from src.graph import Hypergraph


def bfs_din_map(hg: Hypergraph) -> int:
    """
    bfs din map
    """

    adj = hg.get_digraph_adj_list()
    n = len(adj)
    ext = [dict() for _ in range(hg.n)]
    visited = [False] * hg.n
    count = 0

    def update_ext(ext: dict[int, int], elements: Iterable[int]):
        for u in elements:
            if u not in ext:
                ext[u] = 0
            ext[u] += 1

    def bfs(adj: list[list[int]], start_vertex: int):
        count = 0
        queue = deque([start_vertex])
        in_queue = [False] * len(adj)

        visited[start_vertex] = True
        in_queue[start_vertex] = True

        while queue:
            n = queue.popleft()
            visited[n] = True

            for u in adj[n]:
                if not visited[u]:
                    if u in ext[n]:
                        count += ext[u][n]
                    update_ext(ext[u], adj[n])
                if not in_queue[u]:
                    in_queue[u] = True
                    queue.append(u)
        return count

    for i in range(n):
        if not visited[i]:
            count += bfs(adj, i)

    return count
