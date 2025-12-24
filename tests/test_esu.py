from src.loaders import load_hospital
from src.motifs.esu import ad_hoc, esu
from tests.util import Colors, Loader, time_function

hg = load_hospital()
# (76994, 154238, 85214, 33450, 93458, 12982)

assert not hg.has_multiedge()

# print("Running motifs base esu")
# print(f"Found {time_function(lambda: esu(hg, 4))[0]} connected subgraphs")
#
# print()
#
# print("Running ad hoc esu")
# print(f"Found {time_function(lambda: ad_hoc(hg, 4))[0]} connected subgraphs")


def latex_table():
    tests = {
        "hospital": {
            "esu": {
                3,
                4,
            },
            "ad_hoc": {3, 4},
        },
        # "conference": {
        #     "esu": {
        #         3,
        #     },
        #     "ad_hoc": {3, 4},
        # },
    }

    table_body = ""

    for dataset, order in tests.items():
        esu_times = {3: -1.0, 4: -1.0}
        ad_hoc_times = {3: -1.0, 4: -1.0}

        hg = {}
        for o in order["esu"] | order["ad_hoc"]:
            if o not in hg:
                print(f"{Colors.BRIGHT_BLUE}Loading dataset{Colors.RESET}")
                hg[o] = Loader(dataset).order(o).load()

        for o in order["esu"]:
            print("Running esu")
            rv, duration = time_function(lambda: esu(hg[o], o))
            print(f"{Colors.GREEN}Found {rv} connected subgraphs {Colors.RESET}")
            esu_times[o] = duration

        for o in order["ad_hoc"]:
            print("Running ad hoc")
            rv, duration = time_function(lambda: ad_hoc(hg[o], o))
            print(f"{Colors.GREEN}Found {rv} connected subgraphs {Colors.RESET}")
            ad_hoc_times[o] = duration

        esu_times_str = {k: "\\" if v < 0 else f"{int(v * 1000)} ms" for k, v in esu_times.items()}
        ad_hoc_times_str = {k: "\\" if v < 0 else f"{int(v * 1000)} ms" for k, v in ad_hoc_times.items()}

        print(esu_times_str)
        print(ad_hoc_times_str)
        table_body += f"{dataset} & {esu_times_str[3]} & {ad_hoc_times_str[3]} & {esu_times_str[4]} & {ad_hoc_times_str[4]} \\\\\n"

    header = r"""
\begin{center}
	\begin{tabular}{l l l l l}
        \toprule 
        & \multicolumn{2}{l}{Order 3 motifs} & \multicolumn{2}{l}{Order 3 motifs} \\
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


latex_table()
