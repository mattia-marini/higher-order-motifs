import time
from multiprocessing import Manager, Process

from src.loaders import load_hospital
from src.motifs.esu import ad_hoc, esu
from tests.util import Colors, Loader, time_function

hg = load_hospital()

assert not hg.has_multiedge()


def esu_wrapper(dataset, hg, f_type, order, rv_dict):
    esu_f = esu if f_type == "esu" else ad_hoc
    rv, time = time_function(lambda: esu_f(hg, order))
    rv_dict[dataset][f_type][order] = (rv, time)


def run_test(dataset, hg, f_type, order, rv_dict, timeout=5):
    print(f"Starting test: dataset={dataset}, f_type={f_type}, order={order}, timeout={timeout}s")
    process = Process(target=esu_wrapper, args=(dataset, hg, f_type, order, rv_dict))
    process.start()

    process.join(timeout=timeout)

    if process.is_alive():
        # Timeout reached - kill the process
        print(f"Timeout reached for dataset={dataset}, f_type={f_type}, order={order}. Terminating process.")
        process.terminate()
        process.join()
        rv_dict[dataset][f_type][order] = None


def latex_table():
    tests = {
        "hospital": {
            "esu": {
                3: 15,
                4: 15,
            },
            "ad_hoc": {
                3: 15,
                4: 15,
            },
        },
        "conference": {
            "esu": {
                3: 15,
                4: 15,
            },
            "ad_hoc": {
                3: 15,
                4: 15,
            },
        },
        "primary_school": {
            "esu": {
                3: 15,
                4: 15,
            },
            "ad_hoc": {
                3: 15,
                4: 15,
            },
        },
        "high_school": {
            "esu": {
                3: 15,
                4: 15,
            },
            "ad_hoc": {
                3: 15,
                4: 15,
            },
        },
    }

    table_body = ""

    # Loading graphs
    hypergraphs = {}
    for dataset, desc in tests.items():
        hypergraphs[dataset] = {}
        for o in set(desc["esu"].keys()) | set(desc["ad_hoc"].keys()):
            if o not in hypergraphs[dataset]:
                print(f"{Colors.BRIGHT_BLUE}Loading graph for dataset {dataset}, order {o} {Colors.RESET}")
                hypergraphs[dataset][o] = Loader(dataset).order(o).load()

    # Running tests in parallel with timeouts
    processes = []
    manager = Manager()
    times = manager.dict()
    for dataset, desc in tests.items():
        times[dataset] = manager.dict()
        times[dataset]["esu"] = manager.dict()
        times[dataset]["ad_hoc"] = manager.dict()

        for order, timeout in desc["esu"].items():
            p = Process(target=run_test, args=(dataset, hypergraphs[dataset][order], "esu", order, times, timeout))
            p.start()
            processes.append(p)

        for order, timeout in desc["ad_hoc"].items():
            p = Process(target=run_test, args=(dataset, hypergraphs[dataset][order], "ad_hoc", order, times, timeout))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    # print(processes)
    for dataset, f_types in times.items():
        print(dataset)
        for f_type, orders in f_types.items():
            print(f"\t{f_type}:\n\t\t{orders}")

    # esu_times = {3: -1.0, 4: -1.0}
    # ad_hoc_times = {3: -1.0, 4: -1.0}
    #
    #
    # esu_times_str = {k: "\\" if v < 0 else f"{int(v * 1000)} ms" for k, v in esu_times.items()}
    # ad_hoc_times_str = {k: "\\" if v < 0 else f"{int(v * 1000)} ms" for k, v in ad_hoc_times.items()}
    #
    # print(esu_times_str)
    # print(ad_hoc_times_str)
    for dataset, f_types in times.items():
        t = [""] * 4
        t[0] = f_types["esu"][3]
        t[1] = f_types["ad_hoc"][3]
        t[2] = f_types["esu"][4]
        t[3] = f_types["ad_hoc"][4]
        t[0] = f"$>$ {tests[dataset]['esu'][3]} s" if t[0] is None else f"{int(t[0][1] * 1000)} ms"
        t[1] = f"$>$ {tests[dataset]['ad_hoc'][3]} s" if t[1] is None else f"{int(t[1][1] * 1000)} ms"
        t[2] = f"$>$ {tests[dataset]['esu'][4]} s" if t[2] is None else f"{int(t[2][1] * 1000)} ms"
        t[3] = f"$>$ {tests[dataset]['ad_hoc'][4]} s" if t[3] is None else f"{int(t[3][1] * 1000)} ms"
        table_body += f"{dataset} & {t[0]} & {t[1]} & {t[2]} & {t[3]} \\\\\n"

    header = r"""
\begin{center}
	\begin{tabular}{l l l l l}
        \toprule 
        & \multicolumn{2}{l}{Order 3 motifs} & \multicolumn{2}{l}{Order 4 motifs} \\
        \cmidrule(lr){2-3} \cmidrule(lr){4-5}
        Dataset & Esu & Ad-hoc & Esu & Ad-hoc \\
        \midrule
    """
    footer = r"""
        \bottomrule
	\end{tabular}
\end{center}
    """

    print(f"{header}{table_body}{footer}")


if __name__ == "__main__":
    latex_table()
