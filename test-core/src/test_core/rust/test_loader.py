import os

import python_core as pc
from python_core.graph import StandardConstructionMethod
from python_core.loaders import load_pacs
from rich.align import Align
from rich.columns import Columns
from rich.panel import Panel
import rust_core as rc

from rich.table import Table
from rich.console import Console
console = Console()

from test_core.util import  time_function, time_function_p


def run() -> None:
    # test_conference_uw()
    # test_conference_w()
    #
    # test_primary_school_uw()
    # test_primary_school_w()
    #
    # test_high_school_uw()
    # test_high_school_w()
    #
    # test_hospital_uw()
    # test_hospital_w()
    #
    # test_facebook_hs_uw()
    #
    # test_friendship_hs_uw()
    # test_friendship_hs_w()
    #
    # test_gene_disease_w()

    test_pacs_uw()
    test_pacs_w()






def test_conference_uw():
    console.print(test_loader(
        "Conference Unweighted",
        lambda: pc.loaders.load_conference(StandardConstructionMethod(weighted=False)),
        lambda: rc.loader.load_conference_uw(dataset_path("conference.dat"), None)
    ))

def test_conference_w():
    console.print(test_loader(
        "Conference Weighted",
        lambda: pc.loaders.load_conference(StandardConstructionMethod(weighted=True)),
        lambda: rc.loader.load_conference_w(dataset_path("conference.dat"), None)
    ))

def test_primary_school_uw():
    console.print(test_loader(
        "Primary School Unweighted",
        lambda: pc.loaders.load_primary_school(StandardConstructionMethod(weighted=False)),
        lambda: rc.loader.load_primary_school_uw(dataset_path("primaryschool.csv"), None)
    ))

def test_primary_school_w():
    console.print(test_loader(
        "Primary School Weighted",
        lambda: pc.loaders.load_primary_school(StandardConstructionMethod(weighted=True)),
        lambda: rc.loader.load_primary_school_w(dataset_path("primaryschool.csv"), None)
    ))

def test_high_school_uw():
    console.print(test_loader(
        "High School Unweighted",
        lambda: pc.loaders.load_high_school(StandardConstructionMethod(weighted=False)),
        lambda: rc.loader.load_high_school_uw(dataset_path("High-School_data_2013.csv"), None)
    ))

def test_high_school_w():
    console.print(test_loader(
        "High School Weighted",
        lambda: pc.loaders.load_high_school(StandardConstructionMethod(weighted=True)),
        lambda: rc.loader.load_high_school_w(dataset_path("High-School_data_2013.csv"), None)
    ))

def test_hospital_uw():
    console.print(test_loader(
        "Hospital Unweighted",
        lambda: pc.loaders.load_hospital(StandardConstructionMethod(weighted=False)),
        lambda: rc.loader.load_hospital_uw(dataset_path("hospital.dat"), None)
    ))

def test_hospital_w():
    console.print(test_loader(
        "Hospital Weighted",
        lambda: pc.loaders.load_hospital(StandardConstructionMethod(weighted=True)),
        lambda: rc.loader.load_hospital_w(dataset_path("hospital.dat"), None)
    ))

def test_facebook_hs_uw():
    console.print(test_loader(
        "Facebook High School Unweighted",
        lambda: pc.loaders.load_facebook_hs(StandardConstructionMethod(weighted=False)),
        lambda: rc.loader.load_facebook_hs(dataset_path("Facebook-known-pairs_data_2013.csv"), None)
    ))

def test_friendship_hs_uw():
    console.print(test_loader(
        "Friendship High School Unweighted",
        lambda: pc.loaders.load_friendship_hs(StandardConstructionMethod(weighted=False)),
        lambda: rc.loader.load_friendship_hs_uw(dataset_path("Friendship-network_data_2013.csv"), None)
    ))

def test_friendship_hs_w():
    console.print(test_loader(
        "Friendship High School Weighted",
        lambda: pc.loaders.load_friendship_hs(StandardConstructionMethod(weighted=True)),
        lambda: rc.loader.load_friendship_hs_w(dataset_path("Friendship-network_data_2013.csv"), None)
    ))

def test_gene_disease_w():
    console.print(test_loader(
        "Gene Disease Weighted",
        lambda: pc.loaders.load_gene_disease(StandardConstructionMethod(weighted=True, limit_edge_size=10)),
        lambda: rc.loader.load_gene_disease(dataset_path("curated_gene_disease_associations.tsv"), None)
    ))

def test_pacs_uw():
    console.print(test_loader(
        "PACS Unweighted",
        lambda: pc.loaders.load_pacs(StandardConstructionMethod(weighted=False)),
        lambda: rc.loader.load_pacs_uw(dataset_path("PACS.csv"), None)
    ))

def test_pacs_w():
    console.print(test_loader(
        "PACS Weighted",
        lambda: pc.loaders.load_pacs(StandardConstructionMethod(weighted=True)),
        lambda: rc.loader.load_pacs_w(dataset_path("PACS.csv"), None)
    ))

def dataset_path(name):
    return os.path.join(os.environ["DATASET_DIR"], name)



def test_loader(title, python_function, rust_function): 
    # 1. Execute loaders and get timings/graphs
    hg_rust, rust_time = time_function(lambda: rust_function())
    hg_py, python_time = time_function(lambda: python_function())

    # 2. Left Table: Overall Comparison Summary (Row-Oriented Layout)
    summary_table = Table(title="Summary")
    # Columns are bolded/neutral, colors are applied horizontally per row instead
    summary_table.add_column("attr", style="bold")
    summary_table.add_column("Python")
    summary_table.add_column("Rust")
    
    # Format time beautifully
    py_time_str = f"{python_time:.4f}s" if isinstance(python_time, (int, float)) else str(python_time)
    rust_time_str = f"{rust_time:.4f}s" if isinstance(rust_time, (int, float)) else str(rust_time)

    # Populate rows with horizontal colors (Row 1: violet, Row 2: green)
    summary_table.add_row("#edge", str(hg_py.m), str(hg_rust.m()), style="violet")
    summary_table.add_row("time", py_time_str, rust_time_str, style="green")

    # 3. Right Table: Hypergraph Order Comparison
    order_table = Table(title="Hg order comparison")
    order_table.add_column("Hg order")
    order_table.add_column("Rust count")
    order_table.add_column("Python count")
    
    for i in range(2, 6):
        rust_count = hg_rust.count(i)
        py_count = len(hg_py.get_order_map().get(i, []))
        order_table.add_row(f"h{i}", str(rust_count), str(py_count))

    # 4. Place tables side-by-side using Columns, centering the group
    side_by_side = Columns([summary_table, order_table], align="center", expand=False)
    
    # Centering content relative to total terminal width
    centered_content = Align.center(side_by_side)

    # 5. Wrap everything inside your Panel
    panel = Panel(
        renderable=centered_content,
        title=title,
        expand=False  # Keeps the panel tight around the tables
    )
    
    return panel
