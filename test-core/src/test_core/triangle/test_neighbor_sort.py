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
from python_core.triangle.common import (
    degeneracy_ordering,
    sort_adj_list,
    sort_adj_list_bucket,
    sort_adj_list_rust,
)
from rich.console import Console

from test_core.util import load_and_time, time_function_p

console = Console()


def run():
    hg = load_and_time(lambda: load_PACS(StandardConstructionMethod(weighted=True)))

    adj1 = hg.get_digraph_adj_list()
    adj2 = hg.get_digraph_adj_list()
    adj3 = hg.get_digraph_adj_list()

    console.print(hg)

    _, _ = time_function_p(lambda: sort_adj_list_bucket(adj1))  # its in place
    adj2, _ = time_function_p(lambda: sort_adj_list(adj2))
    adj3, _ = time_function_p(lambda: sort_adj_list_rust(adj3))

    assert adj1 == adj2 == adj3, "Sorted adjacency lists do not match"
