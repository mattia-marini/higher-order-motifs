import os
import time
from typing import cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import python_core.loaders as loaders
from python_core.graph import *
from python_core.motifs.motifs2 import motifs_order_3, motifs_order_4
from python_core.motifs.motifs_count_base import generate_motifs
from rich.console import Console

console = Console()


class Loader:
    def __init__(self, dataset):
        self._dataset: str = dataset
        self._construction_method: ConstructionMethodBase = StandardConstructionMethod()

    def construction_method(self, construction_method: ConstructionMethodBase):
        self._construction_method = construction_method
        return self

    def load(self) -> Hypergraph:
        console.print(f'[green]Loading [bold]"{self._dataset}"[/bold] dataset[/]')

        # Dynamically call the appropriate loader function
        loader_name = f"load_{self._dataset}"
        loader = getattr(loaders, loader_name, None)
        if not loader:
            raise ValueError(f"Loader function '{loader_name}' not found in app.loaders")

        hg = cast(Hypergraph, loader(self._construction_method))
        # hg.compute_adjacency()

        console.print(f"[green]Loaded hypergraph for dataset [bold]{self._dataset}[/]")

        return hg


def load_and_time(func, *args, **kwargs):
    def guess_name_from_lambda(fn):
        # only for simple lambdas of the form: lambda: SOME_GLOBAL(...)
        co = getattr(fn, "__code__", None)
        if co is None:
            return None
        names = co.co_names  # tuple of names referenced by the code object
        if not names:
            return None
        # heuristics: the first name is often the callee
        callee_name = names[0]
        # try to resolve the name in the lambda's globals
        candidate = fn.__globals__.get(callee_name)
        if candidate is not None:
            return getattr(candidate, "__name__", callee_name)

        return callee_name

    func_name = guess_name_from_lambda(func)

    dataset = "" if func_name == None else func_name.split("_")[1]  # Assuming function name is like "load_datasetname"
    console.print(f"[green]Loading [bold]{dataset}[/bold] dataset[/]")

    start = time.time()
    rv = func()
    end = time.time()
    elapsed = end - start
    console.print(f"[green]Loaded hypergraph for dataset [bold]{dataset} in {elapsed:.4f} s[/]")

    return rv


def time_function_p(func, *args, **kwargs):
    """
    Times how long 'func' takes to run with the provided arguments.
    Returns (result, elapsed_time).
    """
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    console.print(f"[bright cyan]Time elapsed: {end - start:.4f} seconds [/]")
    elapsed = end - start
    return result, elapsed


def time_function(func, *args, **kwargs):
    """
    Times how long 'func' takes to run with the provided arguments.
    Returns (result, elapsed_time).
    """
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    elapsed = end - start
    return result, elapsed


def display_graph(hg: Hypergraph):
    # G = nx.karate_club_graph()  # example graph

    edges = [handle.nodes for handle in hg.get_order_map().get(2, [])]
    G = nx.from_edgelist(edges, create_using=nx.Graph())

    # pos = nx.spring_layout(G, seed=42)  # force-directed layout
    nx.draw(G, with_labels=False, node_size=30, font_size=8, width=0.1)
    plt.title("NetworkX + Matplotlib")
    plt.show()


class Colors:
    RESET = "\033[0m"

    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    REVERSE = "\033[7m"
