# import rust_core.triangle.forward as rc_forward

# from rust_core.triangle import forward as rc_forward
from .common import degree_ordering


def forward(adj: list[list[int]], sort_degrees: bool = False) -> int:
    """
    forward triangle counting
    """
    n = len(adj)
    a = [set() for _ in range(n)]

    def sort() -> int:
        count = 0
        order, pos, _ = degree_ordering(adj)

        for i in range(n):
            u = order[i]
            for v in adj[u]:
                if pos[u] < pos[v]:
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


def forward_hashed(adj: list[list[int]], sort_degrees: bool = False) -> int:
    """
    forward hashed / compact forward triangle counting
    """
    n = len(adj)
    a = [set() for _ in range(n)]

    def sort() -> int:
        count = 0
        hash = [False] * n
        order, pos, _ = degree_ordering(adj)

        for i in range(n):
            u = order[i]
            for v in adj[u]:
                if pos[u] < pos[v]:
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


def forward_rust(adj: list[list[int]], sort_degrees: bool = False) -> int:
    """
    forward triangle counting with rust implementation
    """
    return rc_forward.forward(adj, sort_degrees)


def forward_hashed_rust(adj: list[list[int]], sort_degrees: bool = False) -> int:
    """
    forward hashed / compact forward triangle counting with rust implementation
    """
    return rc_forward.forward_hashed(adj, sort_degrees)


def forward_hcbs_rust(adj: list[list[int]], sort_degrees: bool = False) -> int:
    """
    forward hashed / compact forward triangle counting with rust implementation and cbst neighbors merge
    """
    return rc_forward.forward_hbs(adj, sort_degrees)


def forward_hashed_cloj_rust(adj: list[list[int]], sort_degrees: bool = False) -> int:
    """
    forward hashed / compact forward triangle counting with rust implementation
    """
    return rc_forward.forward_hashed_cloj(adj, sort_degrees)
