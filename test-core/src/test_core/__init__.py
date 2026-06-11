# import os
#
# from test_core.util import time_function_p
#
# from .triangle import test_degree_ordering, test_neighbor_sort, test_triangle_counting
# import os

# from .rust import test_loader


# from test_core.rust import test_clique, test_loader, test_motifs3
from test_core.triangle import test_triangle_counting2


def main() -> None:
    test_triangle_counting2.run()
    # print(len(generate_motifs(4)[1]))
    # test_motifs3.run()
    # test_loader.run()
    # test_clique.run()

    # rc.loader.load_wiki_talk(os.environ["dataset"])
    # edges2 = [(1, 2) for _ in range(100_000)]
    # edges3 = [(1, 2, 3) for _ in range(100_000)]
    # edges4 = [(1, 2, 3, 4) for _ in range(100_000)]

    # time_function_p(lambda: rc.motifs.count_motifs_3((edges2, edges3)))
    # time_function_p(lambda: rc.motifs.count_motifs_4((edges2, edges3, edges4)))

    # x = rc.graph.WeightedHypergraph()
    # x = rc.graph.WeightedHypergraph()
    # x.insert_hx_tuple(((1,2,3),1))
    # x.insert_hx((1,2,3),1)
    # x.insert_hx((3, 2),1)
    # x.insert_hx((9,2,1),1)
    # x.insert_hx((2,1),1)

    # x.insert_hx_tuple(((2,1,3,5),42))
    # x.insert_hx((2,1,33),1)

    # print(x.n())
    # print(x.edges())

    # hg = rc.graph.WeightedHypergraph()
    # print(hg.n)

    # hg1 = rc.graph.UnweightedHypergraph()
    # hh2 = rc.loader.load_wiki_talk("prova", os.environ["dataset"])
    # hg1.count_5()

    # rc.graph.bfs(hg, 10)

    # test_degree_ordering.run()
    # test_triangle_counting.run()

    # test_loader.run()
    # test_neighbor_sort.run()
