from python_core.graph import StandardConstructionMethod
from python_core.loaders import load_hospital
from python_core.triangle.forward import forward
from python_core.triangle.orca import orca
from rich.console import Console
from rich.table import Table

from test_core.util import time_function

console = Console()

construction_method = StandardConstructionMethod(weighted=True)


def run():
    hg = load_hospital(construction_method)
    curr_result = []
    adj = hg.get_digraph_adj_list()

    triangles, elapsed = time_function(lambda: forward(adj, sort_degrees=False))
    curr_result.append(("forward", triangles, elapsed))

    triangles, elapsed = time_function(lambda: orca(adj))
    curr_result.append(("orca", triangles, elapsed))

    table = Table(title=f"[green]Hospital: n = {hg.n}, m = {hg.m}[/]")
    table.add_column("Algorithm", no_wrap=True)
    table.add_column("Count")
    table.add_column("Time (s)")

    for alg, count, elapsed in curr_result:
        table.add_row(alg, str(count), f"{elapsed:.4f}")

    console.print(table)
