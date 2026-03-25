from collections import deque
from typing import Iterable

from src.graph import Hyperedge, Hypergraph
from src.motifs.motifs3 import count_motifs as count_motifs3
from src.triangle import *
from tests.util import Colors, Loader, StandardConstructionMethod, time_function

hg = Hypergraph()
# hg.add_edge(Hyperedge((0, 1)))
# hg.add_edge(Hyperedge((1, 2)))
# hg.add_edge(Hyperedge((2, 3)))
# hg.add_edge(Hyperedge((0, 1)))
# hg.add_edge(Hyperedge((1, 2)))
# hg.add_edge(Hyperedge((2, 0)))
# hg.add_edge(Hyperedge((2, 3)))
# hg.add_edge(Hyperedge((0, 3)))
# hg.add_edge(Hyperedge((1, 3)))

hg = Loader("primary_school").construction_method(StandardConstructionMethod(weighted=True)).load()
hg = hg.filter_orders([2], retain=True)


# def count3(hg: Hypergraph) -> int:
#     adj = hg.get_digraph_adj_list()
#     ext = [{} for _ in range(hg.n)]
#     visited = [False for _ in range(hg.n)]
#     count = 0
#
#     # def compute_ext_rec(n: int, visited: list[bool]):
#     #     nonlocal ext
#     #     visited[n] = True
#     #
#     #     new_ext = set(adj[n]) - {n}
#     #     for u in adj[n]:
#     #         ext[u].update(new_ext - {u})
#     #
#     #     for u in adj[n]:
#     #         if not visited[u]:
#     #             compute_ext_rec(u, visited)
#     #
#     # def count3_rec(n: int, visited: list[bool]):
#     #     nonlocal count
#     #     visited[n] = True
#     #
#     #     new_ext = set(adj[n]) - {n}
#     #     for u in adj[n]:
#     #         if u in ext[n]:
#     #             # print(n)
#     #             count += 1
#     #
#     #     for u in adj[n]:
#     #         if not visited[u]:
#     #             count3_rec(u, visited)
#     def fin():
#         nonlocal count
#         for n in range(hg.n):
#             for u in adj[n]:
#                 if u in ext[n]:
#                     count += 1
#             print(count)
#
#     # compute_ext_rec(0, visited.copy())
#     # print(ext)
#     # count3_rec(0, visited)
#     fin()
#     return count


# print("prova")
motifs3, elapsed = time_function(lambda: count_motifs3(hg, 3))
triangles = motifs3[((1, 2), (1, 3), (2, 3))].count
print(f"count_motifs3:            \t{triangles} triangles \t {elapsed:.4f} seconds")

triangles, elapsed = time_function(lambda: forward(hg, sort_degrees=False))
print(f"forward:                 \t{triangles} triangles \t {elapsed:.4f} seconds")

triangles, elapsed = time_function(lambda: forward(hg, sort_degrees=True))
print(f"forward deg sort:       \t{triangles} triangles \t {elapsed:.4f} seconds")

triangles, elapsed = time_function(lambda: forward(hg, sort_degrees=False))
print(f"compact forward:        \t{triangles} triangles \t {elapsed:.4f} seconds")

triangles, elapsed = time_function(lambda: forward(hg, sort_degrees=True))
print(f"compact forward deg sort:\t{triangles} triangles \t {elapsed:.4f} seconds")

triangles, elapsed = time_function(lambda: cetc(hg))
print(f"cetc:                     \t{triangles} triangles \t {elapsed:.4f} seconds")

triangles, elapsed = time_function(lambda: cetc_s(hg))
print(f"cetc_s:                   \t{triangles} triangles \t {elapsed:.4f} seconds")

triangles, elapsed = time_function(lambda: bfs_din_map(hg))
print(f"t2:                       \t{triangles} triangles \t {elapsed:.4f} seconds")

# print(triangles)
