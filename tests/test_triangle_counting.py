from collections import deque
from typing import Iterable

from rich.console import Console
from rich.panel import Panel
from rich.table import Row, Table

from src.graph import Hyperedge, Hypergraph
from src.loaders import load_justice_ideo
from src.motifs.motifs3 import count_motifs as count_motifs3
from src.triangle import *
from tests.util import Colors, Loader, StandardConstructionMethod, display_graph, load_and_time, time_function

console = Console()

hg = load_and_time(lambda: load_justice_ideo(StandardConstructionMethod(weighted=True))[0])

hg = hg.filter_orders([2], retain=True)

nodes = list(hg.nodes)
console.print(f"[dim]Removed {hg.remove_self_loops()} self loops")


table1 = Table()
order_map = sorted(hg.get_order_map().items())
edges_counts = []

for order, edges in order_map:
    table1.add_column(str(order), no_wrap=True)
    edges_counts.append(str(len(edges)))


table1.add_row(*edges_counts)
console.print(table1)


table2 = Table(title=f"[green]n= {hg.n}, m: {hg.e}[/]")
table2.add_column("Algorithm", no_wrap=True)
table2.add_column("Count")
table2.add_column("Time (s)")


motifs3, elapsed = time_function(lambda: count_motifs3(hg, 3))
triangles = motifs3[((1, 2), (1, 3), (2, 3))].count
table2.add_row("count_motifs3", str(triangles), f"{elapsed:.4f}")

triangles, elapsed = time_function(lambda: forward(hg, sort_degrees=False))
table2.add_row("forward", str(triangles), f"{elapsed:.4f}")

triangles, elapsed = time_function(lambda: forward(hg, sort_degrees=True))
table2.add_row("forward deg sort", str(triangles), f"{elapsed:.4f}")

triangles, elapsed = time_function(lambda: forward(hg, sort_degrees=False))
table2.add_row("compact forward", str(triangles), f"{elapsed:.4f}")

triangles, elapsed = time_function(lambda: forward(hg, sort_degrees=True))
table2.add_row("compact forward deg sort", str(triangles), f"{elapsed:.4f}")

triangles, elapsed = time_function(lambda: cetc(hg))
table2.add_row("cetc", str(triangles), f"{elapsed:.4f}")

triangles, elapsed = time_function(lambda: cetc_s(hg))
table2.add_row("cetc_s", str(triangles), f"{elapsed:.4f}")

triangles, elapsed = time_function(lambda: bfs_din_map(hg))
table2.add_row("bfs din map", str(triangles), f"{elapsed:.4f}")

console.print(table2)

display_graph(hg)
