from python_core.graph import StandardConstructionMethod
from python_core.loaders import (
    load_gene_disease,
)
from python_core.triangle.common import sort_adj_list, sort_adj_list_bucket
from rich.console import Console

from test_core.util import load_and_time, time_function_p


def run():
    console = Console()

    hg = load_and_time(lambda: load_gene_disease(StandardConstructionMethod(weighted=True)))

    adj1 = hg.get_digraph_adj_list()
    adj2 = hg.get_digraph_adj_list()

    console.print(hg)

    time_function_p(lambda: sort_adj_list_bucket(adj1))
    time_function_p(lambda: sort_adj_list(adj2))

    assert adj1 == adj2, "Sorted adjacency lists do not match"
