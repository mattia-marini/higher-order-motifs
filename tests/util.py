from networkx import edges
import _context
import src as app
import pickle
import json
import os
import hashlib
import config as cfg

import _context
import src as app

class TestBuilder:

    motifs_function = {3:app.motifs2.motifs_order_3, 4:app.motifs2.motifs_order_4}
    def __init__(self, dataset):
        self._dataset = dataset  
        self._orders = set() 
        self._normalize_weights = True  
        self._weighted = False  
        self._plot_leading_motifs = False  
        self._plot_dist_motifs = False  
        self._plot_dist_hyperedges_weights = False  
        self._ignore_cache = False  

    def order(self, order):
        if isinstance(order, int):
            order = [order]
        for o in order:
            if o not in [3, 4]:
                raise ValueError("Order must be 3 or 4")
            self._orders.add(o)
        return self

    def dataset(self, dataset):
        self._dataset = dataset
        return self

    def weighted(self, weighted):
        self._weighted = weighted
        return self

    def ingore_cache(self, ignore_cache):
        self._ignore_cache = ignore_cache
        return self
    
    def normalize_weights(self, normalize_weights):
        self._normalize_weights = normalize_weights
        return self

    def with_plots(self, plots):
        if isinstance(plots, str):
            if plots.lower() == 'default':
                self._plot_leading_motifs = True
                self._plot_dist_motifs = False
                self._plot_dist_hyperedges_weights = True
            elif plots.lower() == 'all':
                self._plot_leading_motifs = True
                self._plot_dist_motifs = True
                self._plot_dist_hyperedges_weights = True

        if isinstance(plots, bool):
            self._plot_leading_motifs = plots
            self._plot_dist_motifs = plots
            self._plot_dist_hyperedges_weights = plots
        return self

    def plot_leading_motifs(self, plot):
        self._plot_leading_motifs = plot
        return self

    def plot_dist_motifs(self, plot):
        self._plot_dist_motifs = plot
        return self

    def plot_dist_hyperedges_weights(self, plot):
        self._plot_dist_hyperedges_weights = plot

    def run(self):
        print(f"Loading {Colors.BOLD}\"{self._dataset}\"{Colors.RESET} dataset")

        # Dynamically call the appropriate loader function
        loader_name = f"load_{self._dataset}{"_duplicates" if self._weighted else ""}"
        loader = getattr(app.loaders, loader_name, None)

        if not loader:
            raise ValueError(f"Loader function '{loader_name}' not found in app.loaders")

        # Load data using the loader
        edges = None
        tot = None
        if self._weighted:
            edges, tot = loader(4)
        else:
            edges = loader(4)

        # Plot hyperedge weight distribution if enabled
        if self._weighted and self._plot_dist_hyperedges_weights:
            app.plot_utils.plot_dist_hyperedges_weights(tot, f"{self._dataset}_weight_dist")

        # Normalize weights if enabled
        if self._weighted and self._normalize_weights:
            app.utils.normalize_weights(edges)

        # Analyze motifs
        for order in self._orders:

            # Checking if motifs for a certain dataset are cached
            motifs = self.get_motifs_cached(order, edges)
            
            # Plot motif distributions if enabled
            if self._plot_dist_motifs:
                app.plot_utils.plot_dist_motifs(motifs, f"{self._dataset}_motifs_{order}", 6)

            # Plot leading motifs if enabled
            if self._plot_leading_motifs:
                app.plot_utils.plot_leading_motifs(motifs, f"{self._dataset}_leading_{"weighted" if self._weighted else "unweighted"}_motifs_{order}", 6, limit=10)

        print(f"{Colors.GREEN}Finished running test for dataset {self._dataset}{Colors.RESET}")


    def get_motifs_cached(self, order, edges):
        cache_file = f"{cfg.MOTIFS_CACHE_DIR}/{self._dataset}_{order}{"w" if self._weighted else ""}_{hash_dataset(edges)}.pkl"
        motifs = None

        if os.path.exists(cache_file) and not self._ignore_cache:
            print(f"{Colors.YELLOW}Loading cached motifs{Colors.RESET}")
            with open(cache_file, "rb") as f:
                motifs = pickle.load(f)
        else:
            print(f"{Colors.BLUE}Computing motifs{Colors.RESET}")
            motifs = self.motifs_function[order](edges, weighted=self._weighted)
            os.makedirs(cfg.MOTIFS_CACHE_DIR, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(motifs, f)

        return motifs


def hash_dataset(edges):
    """
    Returns a SHA-1 hash of a dictionary.
    """
    # Sort keys so the order is consistent

    if isinstance(edges, dict):
        data_array = [x for x in edges.items()]
    elif isinstance(edges, set):
        data_array = [x for x in edges]
    else: 
        raise ValueError("Edges should be either set[tuple[int]] or dict[tuple[int], int]")
    encoded = json.dumps(data_array).encode()
    return hashlib.sha1(encoded).hexdigest()


class Colors:
    RESET = "\033[0m"

    # Regular colors
    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    # Bright colors
    BRIGHT_BLACK   = "\033[90m"
    BRIGHT_RED     = "\033[91m"
    BRIGHT_GREEN   = "\033[92m"
    BRIGHT_YELLOW  = "\033[93m"
    BRIGHT_BLUE    = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN    = "\033[96m"
    BRIGHT_WHITE   = "\033[97m"

    # Styles
    BOLD      = "\033[1m"
    DIM       = "\033[2m"
    UNDERLINE = "\033[4m"
    REVERSE   = "\033[7m"
