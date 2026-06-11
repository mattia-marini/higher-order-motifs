from collections import deque
from typing import Iterable

from python_core.graph import Hyperedge, Hypergraph
from python_core.loaders import (
    load_conference,
    load_DBLP,
    load_enron,
    load_example,
    load_friendship_hs,
    load_gene_disease,
    load_hospital,
    load_justice_ideo,
    load_NDC_classes,
    load_NDC_substances,
    load_random_hypergraph,
    load_wiki,
    load_wiki_talk,
)
from python_core.motifs.motifs3 import count_motifs as count_motifs3
from python_core.triangle.cetc import cetc, cetc_rust, cetc_s, cetc_s_rust
from python_core.triangle.common import degeneracy_ordering
from python_core.triangle.forward import (
    forward,
    forward_hashed,
    forward_hashed_cloj_rust,
    forward_hashed_rust,
    forward_hcbs_rust,
    forward_rust,
)
from python_core.triangle.kclist import kclist, kclist_rust
from python_core.triangle.orca import orca
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Row, Table

from test_core.util import (
    Colors,
    Loader,
    StandardConstructionMethod,
    display_graph,
    load_and_time,
    time_function,
    time_function_p,
)

console = Console()

construction_method = StandardConstructionMethod(weighted=True)


def run():
    graphs = {
        # "random": load_and_time(lambda: load_random_hypergraph(100, 200)),
        "hospital": load_and_time(lambda: load_hospital(construction_method)),
        # "conference": load_and_time(lambda: load_conference(construction_method)),
        # "wiki": load_and_time(lambda: load_wiki(construction_method)),
        # Bigger datasets
        # "PACS": load_and_time(lambda: load_PACS(construction_method)),
        # "ndc_substances": load_and_time(lambda: load_NDC_substances(construction_method)),
        # "DBLP": load_and_time(lambda: load_DBLP(construction_method)),
        # "enron": load_and_time(lambda: load_enron(construction_method)),
        # "gene_disease": load_and_time(lambda: load_gene_disease(construction_method)),
        # "wiki-talk": load_and_time(lambda: load_wiki_talk(construction_method)),
    }

    for name, graph in graphs.items():
        console.print(graph)

    for name, graph in graphs.items():
        graph = graph.filter_orders([2], retain=True)
        graphs[name] = graph
        console.print(f"[dim]Removed {graph.remove_self_loops()} self loops from {graph}")

    dataset_table = Table()
    dataset_table.add_column("Dataset")
    dataset_table.add_column("nodes")
    dataset_table.add_column("2-edges")

    for name, graph in graphs.items():
        dataset_table.add_row(name, str(graph.n), str(graph.m))

    console.print(dataset_table)

    results = []
    for name, hg in graphs.items():
        assert not hg.has_multiedge()
        assert not hg.has_self_loops()

        table = Table(title=f"[green]{name}: n = {hg.n}, m = {hg.m}[/]")
        table.add_column("Algorithm", no_wrap=True)
        table.add_column("Count")
        table.add_column("Time (s)")

        curr_result = []
        adj = hg.get_digraph_adj_list()

        # PYTHON
        triangles, elapsed = time_function(lambda: forward(adj, sort_degrees=False))
        curr_result.append(("forward", triangles, elapsed))

        triangles, elapsed = time_function(lambda: forward(adj, sort_degrees=True))
        curr_result.append(("forward deg sort", triangles, elapsed))

        triangles, elapsed = time_function(lambda: forward_hashed(adj, sort_degrees=False))
        curr_result.append(("forward hashed", triangles, elapsed))

        triangles, elapsed = time_function(lambda: forward_hashed(adj, sort_degrees=True))
        curr_result.append(("forward hashed deg sort", triangles, elapsed))

        triangles, elapsed = time_function(lambda: cetc(adj))
        curr_result.append(("cetc", triangles, elapsed))

        triangles, elapsed = time_function(lambda: cetc_s(adj))
        curr_result.append(("cetc_s", triangles, elapsed))

        triangles, elapsed = time_function(lambda: kclist(adj))
        curr_result.append(("kclist", triangles, elapsed))

        triangles, elapsed = time_function(lambda: orca(adj))
        curr_result.append(("orca", triangles, elapsed))

        # RUST
        # triangles, elapsed = time_function(lambda: forward_rust(adj, sort_degrees=False))
        # curr_result.append(("forward rust", triangles, elapsed))
        #
        # triangles, elapsed = time_function(lambda: forward_rust(adj, sort_degrees=True))
        # curr_result.append(("forward deg sort rust", triangles, elapsed))
        #
        # triangles, elapsed = time_function(lambda: forward_hashed_rust(adj, sort_degrees=False))
        # curr_result.append(("forward hashed rust", triangles, elapsed))
        #
        # triangles, elapsed = time_function(lambda: forward_hashed_rust(adj, sort_degrees=True))
        # curr_result.append(("forward hashed deg sort rust", triangles, elapsed))
        #
        # triangles, elapsed = time_function(lambda: forward_hcbs_rust(adj, sort_degrees=False))
        # curr_result.append(("forward hcbs rust", triangles, elapsed))
        #
        # triangles, elapsed = time_function(lambda: forward_hcbs_rust(adj, sort_degrees=True))
        # curr_result.append(("forward hcbs deg sort rust", triangles, elapsed))
        #
        # triangles, elapsed = time_function(lambda: cetc_rust(adj))
        # curr_result.append(("cetc rust", triangles, elapsed))
        #
        # triangles, elapsed = time_function(lambda: cetc_s_rust(adj))
        # curr_result.append(("cetc_s rust", triangles, elapsed))
        #
        # triangles, elapsed = time_function(lambda: kclist_rust(adj))
        # curr_result.append(("kclist rust", triangles, elapsed))
        #
        # triangles, elapsed = time_function(lambda: forward_hashed_cloj_rust(adj))
        # curr_result.append(("forward hadhes cloj rust", triangles, elapsed))
        #
        # algorithms that are too bad to even try
        # motifs3, elapsed = time_function(lambda: count_motifs3(adj, 3))
        # triangles = motifs3[((1, 2), (1, 3), (2, 3))].count
        # current_result.append(("count_motifs3", triangles, elapsed))

        # triangles, elapsed = time_function(lambda: bfs_din_map(adj))
        # curr_result.append(("bfs din map", triangles, elapsed))

        for alg, count, elapsed in curr_result:
            table.add_row(alg, str(count), f"{elapsed:.4f}")

        results.append((name, curr_result))
        console.print(table)

    summary_table = Table(title="[bold magenta]Summary of Triangle Counts and Runtimes[/]")
    summary_table.add_column("Dataset")
    summary_table.add_column("Nodes")
    summary_table.add_column("Edges")
    summary_table.add_column("Density")
    summary_table.add_column("Degeneracy")
    summary_table.add_column("Best algo")
    summary_table.add_column("Count")
    summary_table.add_column("Time (s)")

    summary_rows = []

    for dataset, result in track(results, description="Summarizing results..."):
        winner = min(result, key=lambda x: x[2])

        density = 2 * graphs[dataset].m / (graphs[dataset].n * (graphs[dataset].n - 1))

        degeneracy = degeneracy_ordering(graphs[dataset].get_digraph_adj_list())[2]
        summary_rows.append(
            (
                dataset,
                str(graphs[dataset].n),
                str(graphs[dataset].m),
                f"{density:.4f}",
                str(degeneracy),
                winner[0],
                str(winner[1]),
                f"{winner[2]:.4f}",
            )
        )

    summary_rows.sort(key=lambda x: x[7], reverse=True)  # Sort by time name
    for row in summary_rows:
        summary_table.add_row(*row)
    console.print(summary_table)
