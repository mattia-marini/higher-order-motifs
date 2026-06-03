from rust_core.motifs import count_motifs_4
from test_core.rust.util import dataset_path, cache_dir
from test_core.util import time_function_p


def run():
    from rust_core.loader import load_pacs_uw
    from rust_core.motifs import count_motifs_3_unweighted

    hg, _ = time_function_p(lambda:load_pacs_uw(dataset_path("PACS.csv"), cache_dir()))

    print(hg.count(2))

    # time_function_p(lambda: count_motifs_3_unweighted(hg))
    time_function_p(lambda: count_motifs_4(hg))
