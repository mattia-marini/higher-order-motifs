from python_core.graph import ConstructionMethodBase, StandardConstructionMethod
from python_core.loaders import (
    load_babbuini,
    load_conference,
    load_DBLP,
    load_enron,
    load_gene_disease,
    load_hospital,
    load_NDC_classes,
    load_PACS,
    load_wiki,
    load_workspace,
)
from python_core.triangle.common import degeneracy_ordering, sort_adj_list, sort_adj_list_bucket
from rich.console import Console

from test_core.util import load_and_time, time_function_p

console = Console()


def run():
    hg = load_and_time(lambda: load_gene_disease(StandardConstructionMethod(weighted=True)))

    adj1 = hg.get_digraph_adj_list()
    adj2 = hg.get_digraph_adj_list()

    console.print(hg)

    time_function_p(lambda: sort_adj_list_bucket(adj1))
    time_function_p(lambda: sort_adj_list(adj2))

    assert adj1 == adj2, "Sorted adjacency lists do not match"
