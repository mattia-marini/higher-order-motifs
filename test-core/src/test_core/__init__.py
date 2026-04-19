from test_core.util import time_function_p

from .triangle import test_degree_ordering, test_neighbor_sort, test_triangle_counting


def main() -> None:
    # import rust_core as rc
    # edges2 = [(1, 2) for _ in range(100_000)]
    # edges3 = [(1, 2, 3) for _ in range(100_000)]
    # edges4 = [(1, 2, 3, 4) for _ in range(100_000)]

    # time_function_p(lambda: rc.motifs.count_motifs_3((edges2, edges3)))
    # time_function_p(lambda: rc.motifs.count_motifs_4((edges2, edges3, edges4)))

    # test_degree_ordering.run()
    test_triangle_counting.run()
    # test_neighbor_sort.run()
