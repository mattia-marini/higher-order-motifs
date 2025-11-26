"""
This file implements the efficient algorithm for motif discovery in hypergraphs.
"""

from .loaders import *
from .utils import *


def motifs_order_3(hg):
    # assert_hypergraph(edges, weighted=weighted)
    N = 3
    full, visited = motifs_ho_full(hg, N)
    standard = motifs_standard(hg, N, visited)

    for i in range(len(full)):
        print(full[i][0], full[i][1])
        print(standard[i][0], standard[i][1])

    res = []
    for i in range(len(full)):
        res.append((full[i][0], full[i][1] + standard[i][1]))

    return res


def motifs_order_4(hg):
    # assert_hypergraph(edges, weighted=weighted)
    N = 4
    full, visited = motifs_ho_full(hg, N)
    not_full, visited = motifs_ho_not_full(hg, N, visited)
    standard = motifs_standard(hg, N, visited)

    res = []
    for i in range(len(full)):
        res.append((full[i][0], full[i][1] + not_full[i][1] + standard[i][1]))

    return res


# N = 3
#
# edges = load_high_school(N)
#
# output = {}
#
# if N == 3:
#     output['motifs'] = motifs_order_3(edges, -1)
# elif N == 4:
#     output['motifs'] = motifs_order_4(edges, -1)
#
# print(output['motifs'])
#
# STEPS = len(edges)*10
# ROUNDS = 10
#
# results = []
#
# for i in range(ROUNDS):
#     e1 = hypergraph(edges)
#     e1.MH(label='stub', n_steps=STEPS)
#     if N == 3:
#         m1 = motifs_order_3(e1.C, i)
#     elif N == 4:
#         m1 = motifs_order_4(e1.C, i)
#     results.append(m1)
#
# output['config_model'] = results
#
# delta = diff_sum(output['motifs'], output['config_model'])
# norm_delta = norm_vector(delta)
#
# print(norm_delta)
