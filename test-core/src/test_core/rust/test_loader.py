import os

import rust_core as rc


def run() -> None:
    hg = rc.loader.load_wiki_talk(os.environ["DATASET_DIR"], os.environ.get("CACHE_DIR"))

    print(hg.m())
    hg.add_h2((666, 777))
    print(hg.m())
