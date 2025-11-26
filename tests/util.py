import os
import time
from enum import Enum

import numpy as np

import config as cfg
import src.loaders as loaders
from src.graph import *
from src.motifs2 import motifs_order_3, motifs_order_4


class Loader:
    motifs_function = {3: motifs_order_3, 4: motifs_order_4}

    def __init__(self, dataset):
        self._dataset: str = dataset
        self._order: int = 3
        self._construction_method: ConstructionMethodBase = StandardConstructionMethod()
        self._ignore_cache: bool = False

    def order(self, order):
        if order not in [3, 4]:
            raise ValueError("Order must be 3 or 4")
        self._order = order
        return self

    def construction_method(self, construction_method: ConstructionMethodBase):
        self._construction_method = construction_method
        return self

    def ignore_cache(self, ignore=True):
        self._ignore_cache = ignore
        return self

    def load(self) -> tuple[Hypergraph, Any]:
        print(f'Loading {Colors.BOLD}"{self._dataset}"{Colors.RESET} dataset')

        # Dynamically call the appropriate loader function
        loader_name = f"load_{self._dataset}"
        loader = getattr(loaders, loader_name, None)
        if not loader:
            raise ValueError(
                f"Loader function '{loader_name}' not found in app.loaders"
            )

        # Load data using the loader
        # edges = None
        # tot = None
        # if self._weight_type == WeightType.STANDARD:
        #     edges, tot = loader(4)
        # else:
        #     edges = loader(4)

        hg = loader(self._construction_method)

        motifs = self.get_motifs_cached(self._order, hg)

        print(f"{Colors.GREEN}Fetched motifs for dataset {self._dataset}{Colors.RESET}")

        return hg, motifs

    def get_motifs_cached(self, order: int, hg: Hypergraph):
        cache_file = f"{cfg.MOTIFS_CACHE_DIR}/{self._dataset}_{order}_{self._construction_method.description()}_{hg.hash()}.npz"
        motifs = None

        if os.path.exists(cache_file) and not self._ignore_cache:
            print(f"{Colors.YELLOW}Loading cached motifs{Colors.RESET}")
            motifs = self.load_cache(cache_file)
        else:
            # raise NotImplementedError("Motif computation is disabled.")
            print(f"{Colors.BLUE}Computing motifs{Colors.RESET}")
            motifs = self.motifs_function[order](hg)
            os.makedirs(cfg.MOTIFS_CACHE_DIR, exist_ok=True)
            self.save_cache(motifs, cache_file)

        return motifs

    def save_cache(self, motifs, filename):
        motifs = [x[1] for x in motifs]  # Extract only the motifs instances
        flat = np.array(
            [node for motif in motifs for instance in motif for node in instance],
            dtype=np.int32,
        )
        infos = np.array(
            [(len(motif), 0 if len(motif) == 0 else len(motif[0])) for motif in motifs],
            dtype=np.int32,
        )
        np.savez(filename, motifs=flat, infos=infos)

    def load_cache(self, filename):
        data = np.load(filename)

        infos = data["infos"]
        data = data["motifs"]

        idx = 0
        rv = []
        for count, size in infos:
            motif = []
            for _ in range(count):
                motif_instance = []

                for _ in range(size):
                    motif_instance.append(data[idx].item())
                    idx += 1

                motif.append(tuple(motif_instance))
            rv.append(motif)

        return rv


def time_function(func, *args, **kwargs):
    """
    Times how long 'func' takes to run with the provided arguments.
    Returns (result, elapsed_time).
    """
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(
        Colors.DIM
        + Colors.BRIGHT_CYAN
        + f"Time elapsed: {end - start:.4f} seconds"
        + Colors.RESET
    )
    elapsed = end - start
    return result, elapsed


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
