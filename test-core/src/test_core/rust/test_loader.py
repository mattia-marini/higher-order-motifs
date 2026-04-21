import os

import rust_core as rc


def run() -> None:
    rc.loader.load_wiki_talk(os.environ["DATASET_DIR"], os.environ.get("CACHE_DIR"))
