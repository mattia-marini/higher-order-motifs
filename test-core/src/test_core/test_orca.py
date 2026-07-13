from python_core.graph import StandardConstructionMethod
from python_core.loaders import load_conference, load_hospital, load_pacs
from python_core.motifs.motifs3 import count_3, count_4
import python_core.motifs.motifs2 as motifs2
from rich.console import Console
from rust_core.graph import DatasetLoader
from rust_core.motifs import orca_3, orca_4

from test_core.util import time_function

console = Console()


def run():
    test_hospital(3)
    test_hospital(4)
    # test_conference()
    # test_pacs()


def test_hospital(order: int):
    py_loader = load_hospital(StandardConstructionMethod(weighted=True))
    rust_loader = DatasetLoader.builder().hospital().weighted().cached(True).load()

    py_loader = py_loader.filter_orders([2]);
    print(f"Hash multiedges: {py_loader.has_multiedge()}")
    print(f"Hash self loops: {py_loader.has_self_loops()}")

    if order == 3:
        test_dataset_order_3(py_loader, rust_loader)
    elif order == 4:
        test_dataset_order_4(py_loader, rust_loader)


def test_conference():
    py_loader = load_conference(StandardConstructionMethod(weighted=True))
    rust_loader = DatasetLoader.builder().conference().weighted().cached(True).load()
    test_dataset_order_3(py_loader, rust_loader)


def test_pacs():
    py_loader = load_pacs(StandardConstructionMethod(weighted=True))
    rust_loader = DatasetLoader.builder().pacs().weighted().cached(True).load()
    test_dataset_order_3(py_loader, rust_loader)


def test_dataset_order_3(py_hg, rust_hg):
    py_rv_3, py_time_3 = time_function(lambda: count_3(py_hg))
    # py_rv_4, py_time_4 = time_function(lambda: count_4(py_hg))

    order_map = py_hg.get_order_map()
    print("Python")
    print(f"2: {len(order_map.get(2, []))}")
    print(f"3: {len(order_map.get(3, []))}")

    print("Rust")
    print(f"2: {rust_hg.count(2)}")
    print(f"3: {rust_hg.count(3)}")

    print(f"Python: {py_time_3}")
    for motif, stats in py_rv_3.items():
        print(f"Motif {motif}: {stats}")

    rust_rv_3, rust_time_3 = time_function(lambda: orca_3(rust_hg))
    # rust_rv_4, rust_time_4 = time_function(lambda: orca_4(rust_hg))

    print(f"Rust: {rust_time_3}")
    for fingerprint, stats in rust_rv_3.items():
        print(fingerprint.get_canonical_rep())
        print(stats)

    # print(f"Expected 3 motifs: {py_rv_3}, time: {py_time_3:.4f}s")
    # print(f"Rust ORCA 3 motifs: {rust_rv_3}, time: {rust_time_3:.4f}s")


def test_dataset_order_4(py_hg, rust_hg):
    py_rv_3, py_time_3 = time_function(lambda: count_4(py_hg))
    # py_rv_4, py_time_4 = time_function(lambda: count_4(py_hg))

    order_map = py_hg.get_order_map()
    print("Python")
    print(f"2: {len(order_map.get(2, []))}")
    print(f"3: {len(order_map.get(3, []))}")

    print(f"Python: {py_time_3}")
    for motif, stats in py_rv_3.items():
        uniform_2 = True
        for edge in motif:
            if len(edge) != 2:
                uniform_2 = False
                break
        if uniform_2:
            print(f"Motif {motif}: {stats}")

    return
    rust_rv_3, rust_time_3 = time_function(lambda: escape(rust_hg))
    # rust_rv_4, rust_time_4 = time_function(lambda: orca_4(rust_hg))

    print("Rust")
    print(f"2: {rust_hg.count(2)}")
    print(f"3: {rust_hg.count(3)}")

    print(f"Rust: {rust_time_3}; {len(rust_rv_3)} different motifs")
    for fingerprint, stats in rust_rv_3.items():
        print(fingerprint.get_canonical_rep())
        print(stats)

    # print(f"Expected 3 motifs: {py_rv_3}, time: {py_time_3:.4f}s")
    # print(f"Rust ORCA 3 motifs: {rust_rv_3}, time: {rust_time_3:.4f}s")
