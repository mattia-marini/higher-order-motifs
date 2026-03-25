from src.graph import Hypergraph
from src.triangle.common import counting_sort


def forward(hg: Hypergraph, sort_degrees: bool = False) -> int:
    """
    forward triangle counting
    """
    adj = hg.get_digraph_adj_list()

    return forward_raw(adj, sort_degrees)


def forward_raw(adj: list[list[int]], sort_degrees: bool = False) -> int:
    n = len(adj)
    a = [set() for _ in range(n)]

    def sort() -> int:
        count = 0
        degrees = [(i, len(adj[i])) for i in range(n)]
        counting_sort(degrees, key=lambda x: x[1], max_value=n)
        map = [x[0] for x in degrees]
        rank = [0] * n
        for pos, v in enumerate(map):
            rank[v] = pos

        for i in range(n):
            u = map[i]
            for v in adj[u]:
                if rank[u] < rank[v]:
                    for w in a[u].intersection(a[v]):
                        count += 1
                    a[v].add(u)

        return count

    def no_sort() -> int:
        count = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    for w in a[u].intersection(a[v]):
                        count += 1
                    a[v].add(u)
        return count

    if sort_degrees == True:
        return sort()
    else:
        return no_sort()


def compact_forward(hg: Hypergraph, sort_degrees: bool = False) -> int:
    """
    forward hashed / compact forward triangle counting
    """
    a = [set() for _ in range(hg.n)]
    adj = hg.get_digraph_adj_list()

    return compact_forward_raw(adj, sort_degrees)


def compact_forward_raw(adj: list[list[int]], sort_degrees: bool = False) -> int:
    """
    forward hashed / compact forward triangle counting
    """
    n = len(adj)
    a = [set() for _ in range(n)]

    def sort() -> int:
        count = 0
        hash = [False] * n
        degrees = [(i, len(adj[i])) for i in range(n)]
        counting_sort(degrees, key=lambda x: x[1], max_value=n)
        map = [x[0] for x in degrees]
        rank = [0] * n
        for pos, v in enumerate(map):
            rank[v] = pos

        for i in range(n):
            u = map[i]
            for v in adj[u]:
                if rank[u] < rank[v]:
                    for w in a[u]:
                        hash[w] = True
                    for w in a[u].intersection(a[v]):
                        if hash[w]:
                            count += 1
                    for w in a[u]:
                        hash[w] = False
                    a[v].add(u)

        return count

    def no_sort() -> int:
        count = 0
        hash = [False] * n
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    for w in a[u]:
                        hash[w] = True
                    for w in a[u].intersection(a[v]):
                        if hash[w]:
                            count += 1
                    for w in a[u]:
                        hash[w] = False
                    a[v].add(u)
        return count

    if sort_degrees == True:
        return sort()
    else:
        return no_sort()
