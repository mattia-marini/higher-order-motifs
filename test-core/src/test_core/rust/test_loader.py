import os

import python_core as pc
import rust_core as rc

from test_core.util import time_function, time_function_p


def run() -> None:
    conference()


def wiki_talk() -> None:
    hg = rc.loader.load_wiki_talk(os.environ["DATASET_DIR"], os.environ.get("CACHE_DIR"))

    print(hg.m())
    hg.add_h2((666, 777))
    print(hg.m())


def conference() -> None:
    hg, _ = time_function_p(lambda: rc.loader.load_conference(os.environ["DATASET_DIR"], None))

    print(rc.graph.H2T(1, 2))
    print(rc.graph.H3T(1, 2))

    print(hg.count_2())
    print(hg.count_3())
    print(hg.count_4())
    print(f"Rust loaded {hg.m()}")

    hg, _ = time_function_p(lambda: pc.loaders.load_conference())
    # print(hg.get_order_map().get(2, []))

    print(len(hg.get_order_map().get(2, [])))
    print(len(hg.get_order_map().get(3, [])))
    print(len(hg.get_order_map().get(4, [])))
    print(f"Python loaded {hg.m}")
