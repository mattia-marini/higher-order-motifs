from typing import Iterable

from src.graph import Hypergraph


def bfs_din_map(hg: Hypergraph) -> int:
    """
    bfs din map
    """

    adj = hg.get_digraph_adj_list()
    ext = [dict() for _ in range(hg.n)]
    visited = [False] * hg.n
    count = 0
    queue = [0]

    def update_ext(ext: dict[int, int], elements: Iterable[int]):
        for u in elements:
            if u not in ext:
                ext[u] = 0
            ext[u] += 1

    while len(queue) > 0:
        n = queue.pop(0)
        visited[n] = True

        for u in adj[n]:
            if not visited[u]:
                if u in ext[n]:
                    count += ext[u][n]

                update_ext(ext[u], adj[n])
                # count += ext[u]
                if u not in queue:
                    queue.append(u)

        # for u in adj[n]:
        #     ext[u] += 1
        # print(n, ext, count)

    return count
