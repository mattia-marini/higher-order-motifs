from python_core.graph import StandardConstructionMethod
from python_core.loaders import load_hospital
from python_core.motifs.motifs3 import count_3, count_4
from rich.console import Console
from rust_core.graph import DatasetLoader
from rust_core.motifs import orca_3, orca_4

from test_core.util import time_function

console = Console()


def run():
    test_hospital()


def test_hospital():
    py_hg = load_hospital(StandardConstructionMethod(weighted=False))
    py_rv_3, py_time_3 = time_function(lambda: count_3(py_hg))
    # py_rv_4, py_time_4 = time_function(lambda: count_4(py_hg))

    order_map = py_hg.get_order_map()
    print("Python")
    print(f"2: {len(order_map[2])}")
    print(f"3: {len(order_map[3])}")

    rust_hg = DatasetLoader.builder().hospital().unweighted().load()
    rust_rv_3, rust_time_3 = time_function(lambda: orca_3(rust_hg))
    # rust_rv_4, rust_time_4 = time_function(lambda: orca_4(rust_hg))

    print("Rust")
    print(f"2: {rust_hg.count(2)}")
    print(f"3: {rust_hg.count(3)}")
    # x = Fingerprint4(0)

    print(f"Python: {py_time_3}")
    for motif, stats in py_rv_3.items():
        print(f"Motif {motif}: {stats}")

    print(f"Rust: {rust_time_3}")
    for fingerprint, stats in rust_rv_3.items():
        print(fingerprint.get_canonical_rep())
        print(stats)

    # print(f"Expected 3 motifs: {py_rv_3}, time: {py_time_3:.4f}s")
    # print(f"Rust ORCA 3 motifs: {rust_rv_3}, time: {rust_time_3:.4f}s")
