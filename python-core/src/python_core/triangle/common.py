from collections import deque
from typing import Any, Callable


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


def sort_adj_list_bucket(adj: list[list[int]]):
    """
    Bucket sort each adjacency list in-place. Inefficient; sparse graph can end up with O(n^2) complexity
    """
    for neighbors in adj:
        d = len(neighbors)
        if d <= 1:
            continue
        min_v = neighbors[0]
        max_v = neighbors[0]
        for v in neighbors:
            if v < min_v:
                min_v = v
            elif v > max_v:
                max_v = v
        R = max_v - min_v + 1

        # if R > 4 * d:  # heuristic threshold; I made this constant up out of thin air
        #     neighbors.sort()
        #     continue
        buckets = [0] * R
        for v in neighbors:
            buckets[v - min_v] += 1
        idx = 0
        for i in range(R):
            if buckets[i] > 0:
                neighbors[idx] = min_v + i
                idx += 1


def sort_adj_list(adj: list[list[int]]):
    """
    Efficiently sort each adjacency list in-place.
    Time complexity: O(n + m)
    Space complexity: O(n + m)
    """

    n = len(adj)
    sorted = [[] for _ in range(n)]
    for v in range(n):
        for u in adj[v]:
            sorted[u].append(v)

    adj[:] = sorted


def degree_ordering(adj: list[list[int]]) -> tuple[list[int], list[int], int]:
    """
    Returns:
    - order: The degree ordering of vertices
    - pos: Position of vertex i in order (pos[i] gives the index of vertex i in the order list).
    - max_deg: The maximum degree of the graph
    Complexity: O(n)
    """
    n = len(adj)
    if n == 0:
        return [], [], 0

    deg = [len(neighbors) for neighbors in adj]
    max_deg = max(deg) if deg else 0

    # 1. Create bins to count how many nodes have each degree
    bin_count = [0] * (max_deg + 1)
    for d in deg:
        bin_count[d] += 1

    start_pos = 0
    bin_starts = [0] * (max_deg + 1)
    for d in range(max_deg + 1):
        bin_starts[d] = start_pos
        start_pos += bin_count[d]

    order = [0] * n
    pos = [0] * n
    for v in range(n):
        pos[v] = bin_starts[deg[v]]
        order[pos[v]] = v
        bin_starts[deg[v]] += 1

    return order, pos, max_deg


def degeneracy_ordering(adj: list[list[int]]) -> tuple[list[int], list[int], int]:
    """
    Returns a degeneracy ordering of the graph represented by the adjacency list `adj`, along with the position of each vertex in the ordering and the degeneracy of the graph.
    Returns:
    - order: The degeneracy ordering of vertices
    - pos: Position of vertex i in order (pos[i] gives the index of vertex i in the order list).
    - k: The degeneracy of the graph

    Complexity: O(n + m)
    """
    n = len(adj)
    if n == 0:
        return [], [], 0

    deg = [len(neighbors) for neighbors in adj]
    max_deg = max(deg) if deg else 0

    # 1. Create bins to count how many nodes have each degree
    bin_count = [0] * (max_deg + 1)
    for d in deg:
        bin_count[d] += 1

    # 2. Find the starting index for each degree bucket in the 'order' array
    start_pos = 0
    bin_starts = [0] * (max_deg + 1)
    for d in range(max_deg + 1):
        bin_starts[d] = start_pos
        start_pos += bin_count[d]

    # 3. Initial placement of nodes into 'order' and 'pos'
    # We use a copy of bin_starts because we'll move the pointers during fill
    temp_starts = list(bin_starts)
    order = [0] * n
    pos = [0] * n
    for v in range(n):
        pos[v] = temp_starts[deg[v]]
        order[pos[v]] = v
        temp_starts[deg[v]] += 1

    # 4. The main loop: remove node of minimum degree
    k = 0
    for i in range(n):
        v = order[i]
        k = max(k, deg[v])

        for u in adj[v]:
            if pos[u] > i:  # Only look at neighbors still "in the graph"
                u_deg = deg[u]
                u_pos = pos[u]

                # The first node in u's degree bucket
                first_node_pos = bin_starts[u_deg]
                first_node = order[first_node_pos]

                # Swap u with the first node in its bucket to keep buckets contiguous
                if u != first_node:
                    pos[u], pos[first_node] = first_node_pos, u_pos
                    order[u_pos], order[first_node_pos] = first_node, u

                # Move the bucket boundary forward and decrease degree
                bin_starts[u_deg] += 1
                deg[u] -= 1

    return order, pos, k
