# import os
#
# from test_core.util import time_function_p
#
# from .triangle import test_degree_ordering, test_neighbor_sort, test_triangle_counting
import os

from .rust import test_loader


def main() -> None:
    # rc.loader.load_wiki_talk(os.environ["dataset"])
    # edges2 = [(1, 2) for _ in range(100_000)]
    # edges3 = [(1, 2, 3) for _ in range(100_000)]
    # edges4 = [(1, 2, 3, 4) for _ in range(100_000)]

    # time_function_p(lambda: rc.motifs.count_motifs_3((edges2, edges3)))
    # time_function_p(lambda: rc.motifs.count_motifs_4((edges2, edges3, edges4)))
    import rust_core as rc

    hg1 = rc.graph.UnweightedHypergraph()
    # hh2 = rc.loader.load_wiki_talk("prova", os.environ["dataset"])
    hg1.count_5()

    # rc.graph.bfs(hg, 10)

    # test_degree_ordering.run()
    # test_triangle_counting.run()

    # test_loader.run()
    # test_neighbor_sort.run()
