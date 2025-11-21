from networkx import edges
import _context
import numpy as np
import src as app
import pickle
import json
import os
import hashlib
import config as cfg

import _context
import src as app
from enum import Enum


class WeightType(Enum):
    UNWEIGHTED = 1
    STANDARD = 2
    NORMALIZED_DEFAULT = 3
    NORMALIZED_BY_ORDER = 4

class Loader:

    motifs_function = {3:app.motifs2.motifs_order_3, 4:app.motifs2.motifs_order_4}
    def __init__(self, dataset):
        self._dataset = dataset  
        self._order = 3
        self._weight_type = WeightType.UNWEIGHTED
        self._ignore_cache = False  

    def order(self, order):
        if order not in [3, 4]:
            raise ValueError("Order must be 3 or 4")
        self._orders = order
        return self

    def weight_type(self, weight_type):
        if not isinstance(weight_type, WeightType):
            raise ValueError("weight_type must be an instance of WeightType Enum")
        self._weight_type = weight_type
        return self

    def dataset(self, dataset):
        self._dataset = dataset
        return self

    def ingore_cache(self, ignore_cache):
        self._ignore_cache = ignore_cache
        return self
    
    def normalize_weights(self, normalize_weights):
        self._normalize_weights = normalize_weights
        return self

    def ignore_cache(self, ignore = True):
        self._ignore_cache = ignore
        return self

    def load(self):
        print(f"Loading {Colors.BOLD}\"{self._dataset}\"{Colors.RESET} dataset")

        # Dynamically call the appropriate loader function
        loader_name = f"load_{self._dataset}{"_duplicates" if self._weight_type == WeightType.STANDARD else ""}"
        loader = getattr(app.loaders, loader_name, None)

        if not loader:
            raise ValueError(f"Loader function '{loader_name}' not found in app.loaders")

        # Load data using the loader
        edges = None
        tot = None
        if self._weight_type == WeightType.STANDARD:
            edges, tot = loader(4)
        else:
            edges = loader(4)


        motifs = self.get_motifs_cached(self._order, edges)

        print(f"{Colors.GREEN}Fetched motifs for dataset {self._dataset}{Colors.RESET}")

        return edges, motifs


    def get_motifs_cached(self, order, edges):
        cache_file = f"{cfg.MOTIFS_CACHE_DIR}/{self._dataset}_{order}{"w" if self._weight_type != WeightType.UNWEIGHTED else ""}_{hash_dataset(edges)}.npz"
        motifs = None

        if os.path.exists(cache_file) and not self._ignore_cache:
            print(f"{Colors.YELLOW}Loading cached motifs{Colors.RESET}")
            motifs = self.load_cache(cache_file)
        else:
            print(f"{Colors.BLUE}Computing motifs{Colors.RESET}")
            motifs = self.motifs_function[order](edges, weighted=self._weight_type != WeightType.UNWEIGHTED)
            os.makedirs(cfg.MOTIFS_CACHE_DIR, exist_ok=True)
            self.save_cache(motifs, cache_file)

        return motifs


    def save_cache(self, motifs, filename):
        motifs = [x[1] for x in motifs]  # Extract only the motifs instances
        flat = np.array([node for motif in motifs for instance in motif for node in instance], dtype=np.int32)
        infos = np.array([(len(motif), 0 if len(motif) == 0 else len(motif[0])) for motif in motifs], dtype=np.int32)
        np.savez(
            filename,
            motifs=flat,
            infos=infos
        )
    
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
