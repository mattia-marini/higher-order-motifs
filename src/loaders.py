import csv
import random
from collections import defaultdict
from typing import cast

import pandas as pd
from rich.console import Console
from rich.progress import track

import config as cfg
from src.graph import *

console = Console()


# Small datasets
def load_primary_school(
    construction_method: ConstructionMethodBase = StandardConstructionMethod(),
) -> Hypergraph:
    import networkx as nx

    dataset = f"{cfg.DATASET_DIR}/primaryschool.csv"

    fopen = open(dataset, "r")
    lines = fopen.readlines()

    graph = {}
    for l in lines:
        t, a, b, c, d = l.split()
        t = int(t) - 31220
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]

    fopen.close()

    def standard_construction():
        cm = cast(StandardConstructionMethod, construction_method)
        hg = Hypergraph()

        for k in graph.keys():
            e_k = graph[k]
            G = nx.Graph(e_k, directed=False)
            c = list(nx.find_cliques(G))
            for i in c:
                i = tuple(sorted(i))

                if not (cm.limit_edge_size and len(i) > cm.limit_edge_size):
                    if cm.weighted:
                        if hg.has_edge_with_nodes(i):
                            handle = hg.get_first_edges_by_nodes(i)
                            handle.weight += 1.0
                        else:
                            hg.add_edge(Hyperedge(i, 1.0))
                    else:
                        if not hg.has_edge_with_nodes(i):
                            hg.add_edge(Hyperedge(i))

        hg.normalize_weights(cm.normalization_method)
        return hg

    def time_window_construction():
        cm = cast(TimeWindowConstructionMethod, construction_method)
        raise NotImplementedError()

    def temporal_path_construction():
        cm = cast(TemporalPathConstructionMethod, construction_method)
        raise NotImplementedError()

    if isinstance(construction_method, StandardConstructionMethod):
        return standard_construction()
    elif isinstance(construction_method, TimeWindowConstructionMethod):
        return time_window_construction()
    elif isinstance(construction_method, TemporalPathConstructionMethod):
        return temporal_path_construction()
    else:
        raise ValueError("Unknown construction method")


def load_conference(
    construction_method: ConstructionMethodBase = StandardConstructionMethod(),
) -> Hypergraph:
    import networkx as nx

    dataset = f"{cfg.DATASET_DIR}/conference.dat"

    fopen = open(dataset, "r")
    lines = fopen.readlines()

    graph = {}
    for l in lines:
        t, a, b = l.split()
        t = int(t) - 32520
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]

    fopen.close()

    def standard_construction():
        cm = cast(StandardConstructionMethod, construction_method)
        hg = Hypergraph()

        for k in graph.keys():
            e_k = graph[k]
            G = nx.Graph(e_k, directed=False)
            c = list(nx.find_cliques(G))
            for i in c:
                i = tuple(sorted(i))

                if not (cm.limit_edge_size and len(i) > cm.limit_edge_size):
                    if cm.weighted:
                        if hg.has_edge_with_nodes(i):
                            handle = hg.get_first_edges_by_nodes(i)
                            handle.weight += 1.0
                        else:
                            hg.add_edge(Hyperedge(i, 1.0))
                    else:
                        if not hg.has_edge_with_nodes(i):
                            hg.add_edge(Hyperedge(i))

        hg.normalize_weights(cm.normalization_method)
        return hg

    def time_window_construction():
        cm = cast(TimeWindowConstructionMethod, construction_method)
        raise NotImplementedError()

    def temporal_path_construction():
        cm = cast(TemporalPathConstructionMethod, construction_method)
        raise NotImplementedError()

    if isinstance(construction_method, StandardConstructionMethod):
        return standard_construction()
    elif isinstance(construction_method, TimeWindowConstructionMethod):
        return time_window_construction()
    elif isinstance(construction_method, TemporalPathConstructionMethod):
        return temporal_path_construction()
    else:
        raise ValueError("Unknown construction method")


def load_high_school(
    construction_method: ConstructionMethodBase = StandardConstructionMethod(),
) -> Hypergraph:
    import networkx as nx

    dataset = f"{cfg.DATASET_DIR}/High-School_data_2013.csv"

    fopen = open(dataset, "r")
    lines = fopen.readlines()

    graph = {}
    for l in lines:
        t, a, b, c, d = l.split()
        t = int(t) - 1385982020
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]

    fopen.close()

    def standard_construction():
        cm = cast(StandardConstructionMethod, construction_method)
        hg = Hypergraph()

        for k in graph.keys():
            e_k = graph[k]
            G = nx.Graph(e_k, directed=False)
            c = list(nx.find_cliques(G))
            for i in c:
                i = tuple(sorted(i))

                if not (cm.limit_edge_size and len(i) > cm.limit_edge_size):
                    if cm.weighted:
                        if hg.has_edge_with_nodes(i):
                            handle = hg.get_first_edges_by_nodes(i)
                            handle.weight += 1.0
                        else:
                            hg.add_edge(Hyperedge(i, 1.0))
                    else:
                        if not hg.has_edge_with_nodes(i):
                            hg.add_edge(Hyperedge(i))

        hg.normalize_weights(cm.normalization_method)
        return hg

    def time_window_construction():
        cm = cast(TimeWindowConstructionMethod, construction_method)
        raise NotImplementedError()

    def temporal_path_construction():
        cm = cast(TemporalPathConstructionMethod, construction_method)
        raise NotImplementedError()

    if isinstance(construction_method, StandardConstructionMethod):
        return standard_construction()
    elif isinstance(construction_method, TimeWindowConstructionMethod):
        return time_window_construction()
    elif isinstance(construction_method, TemporalPathConstructionMethod):
        return temporal_path_construction()
    else:
        raise ValueError("Unknown construction method")


def load_hospital(
    construction_method: ConstructionMethodBase = StandardConstructionMethod(),
) -> Hypergraph:
    import networkx as nx

    dataset = f"{cfg.DATASET_DIR}/hospital.dat"

    fopen = open(dataset, "r")
    lines = fopen.readlines()

    graph = {}
    for l in lines:
        t, a, b, c, d = l.split()
        t = int(t) - 140
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]

    fopen.close()

    def standard_construction():
        cm = cast(StandardConstructionMethod, construction_method)
        hg = Hypergraph()

        for k in graph.keys():
            e_k = graph[k]
            G = nx.Graph(e_k, directed=False)
            c = list(nx.find_cliques(G))
            for i in c:
                i = tuple(sorted(i))

                if not (cm.limit_edge_size and len(i) > cm.limit_edge_size):
                    if cm.weighted:
                        if hg.has_edge_with_nodes(i):
                            handle = hg.get_first_edges_by_nodes(i)
                            handle.weight += 1.0
                        else:
                            hg.add_edge(Hyperedge(i, 1.0))
                    else:
                        if not hg.has_edge_with_nodes(i):
                            hg.add_edge(Hyperedge(i))

        hg.normalize_weights(cm.normalization_method)
        return hg

    def time_window_construction():
        cm = cast(TimeWindowConstructionMethod, construction_method)
        raise NotImplementedError()

    def temporal_path_construction():
        cm = cast(TemporalPathConstructionMethod, construction_method)
        raise NotImplementedError()

    if isinstance(construction_method, StandardConstructionMethod):
        return standard_construction()
    elif isinstance(construction_method, TimeWindowConstructionMethod):
        return time_window_construction()
    elif isinstance(construction_method, TemporalPathConstructionMethod):
        return temporal_path_construction()
    else:
        raise ValueError("Unknown construction method")

    # plot_dist_hyperedges(tot, "hospital")
    # print(len(edges))
    # count(tot)
    # return edges


# Following datasets are big and could be hard to run locally
def load_facebook_hs(construction_method: ConstructionMethodBase = StandardConstructionMethod(), only_confirmed=True):
    # TODO: generate acual hypergraph
    import pandas as pd

    hg = Hypergraph()

    d = pd.read_csv(
        "{}/Facebook-known-pairs_data_2013.csv".format(cfg.DATASET_DIR), header=None, names=["a", "b", "c"], sep="\\s+"
    )

    for _, row in d.iterrows():
        if not only_confirmed or row.c == 1:
            hg.add_edge(Hyperedge((row.a, row.b)))

    console.print("[yellow][Warning] Ignoring construction method; defaulting to 2-edges only [/]")

    return hg


def load_friendship_hs(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    import pandas as pd

    hg = Hypergraph()

    d = pd.read_csv(
        "{}/Friendship-network_data_2013.csv".format(cfg.DATASET_DIR), header=None, names=["a", "b"], sep="\\s+"
    )

    for _, row in d.iterrows():
        if construction_method.weighted:
            if not hg.has_edge_with_nodes((row.a, row.b)):
                hg.add_edge(Hyperedge((row.a, row.b), 0.0))
            handle = hg.get_first_edges_by_nodes((row.a, row.b))
            handle.weight += 1.0
        else:
            hg.add_edge(Hyperedge((row.a, row.b)))

    console.print("[yellow][Warning] Ignoring construction method; defaulting to 2-edges only [/]")
    return hg


def load_meta_hs(T="sex"):
    tsv_file = open("{}/meta_hs.txt".format(cfg.DATASET_DIR))
    data = csv.reader(tsv_file, delimiter="\t")
    res = {}
    for i in data:
        a, b, c = i
        a = int(a)
        if T == "sex" and c != "Unknown":
            res[a] = c
        elif T == "class":
            res[a] = b

    return res


def load_meta_ps(T="sex"):
    tsv_file = open("{}/metadata_ps.txt".format(cfg.DATASET_DIR))
    data = csv.reader(tsv_file, delimiter="\t")
    res = {}
    for i in data:
        a, b, c = i
        a = int(a)
        if T == "sex" and c != "Unknown":
            res[a] = c
        elif T == "class":
            res[a] = b

    return res


def load_example(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    e = [(1, 2, 3), (2, 4), (2, 3), (2, 5, 6), (4, 6), (1, 2, 3, 7)]
    hg = Hypergraph()
    for i in e:
        if construction_method.limit_edge_size == None or len(i) <= construction_method.limit_edge_size:
            hg.add_edge(Hyperedge(i))
    return hg


def random_hypergraph(N, E):
    hg = Hypergraph()
    for _ in range(E):
        s = random.randint(2, 3)
        hg.add_edge(Hyperedge(random.sample([i for i in range(1, N + 1)], s)))

    return hg


def load_gene_disease(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    name2id_gene = {}
    id_gene2name = {}

    diseases = {}
    idxG = 0

    tsv_file = open("{}/curated_gene_disease_associations.tsv".format(cfg.DATASET_DIR))
    data = pd.read_csv(tsv_file, delimiter="\t", chunksize=1000)
    hg = Hypergraph()

    for chunk in data:
        # chunk is a DataFrame with up to 1000 rows
        for row in chunk.itertuples(index=False):  # row.diseaseId
            # print(row.diseaseName)
            gene = int(row[0])
            dis = row[4]

            if dis not in diseases:
                diseases[dis] = []
            diseases[dis].append(gene)

    for d in diseases.keys():
        if len(diseases[d]) > 1 and len(diseases[d]):
            if not hg.has_edge_with_nodes(diseases[d]):
                hg.add_edge(Hyperedge(diseases[d], 0.0))
            hg.get_edges_by_nodes(diseases[d])[0].weight += 1.0

    return hg
    c = 0
    for row in data:
        c += 1
        if c == 1:
            break
        gene = int(row[0])
        dis = row[4]
        if gene in name2id_gene:
            gene = name2id_gene[gene]
        else:
            name2id_gene[gene] = idxG
            id_gene2name[idxG] = gene
            gene = name2id_gene[gene]
            idxG += 1

        if dis in diseases:
            diseases[dis].append(gene)
        else:
            diseases[dis] = [gene]

    edges = set()
    tot = []

    discarded_1 = 0
    discarded = 0

    for d in diseases.keys():
        if len(diseases[d]) > 1 and len(diseases[d]) <= N:
            edges.add(tuple(sorted(diseases[d])))
        elif len(diseases[d]) == 1:
            discarded_1 += 1
        else:
            discarded += 1

        tot.append(diseases[d])

    tsv_file.close()
    # plot_dist_hyperedges(tot, "gene_disease")
    # print(count(tot))
    return list(edges)


def load_PACS_common():
    import csv

    names = {}  # FullName -> id
    next_name_id = 0
    papers = {}  # ArticleID -> {"authors": [ids], "PACS": pacs}

    # localize builtins for speed
    _names = names
    _papers = papers
    _next_id = next_name_id

    with open(f"{cfg.DATASET_DIR}/PACS.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)  # adjust delimiter if needed, e.g., csv.reader(f, delimiter='\t')
        header = next(reader, None)  # skip header if present
        count = 0
        for row in reader:
            # expect: ArticleID,PACS,FullName
            article_id = row[0]
            pacs = row[1]
            fullname = row[2]

            # map fullname -> id (assign new id if unseen)
            if fullname in _names:
                author_id = _names[fullname]
            else:
                author_id = _next_id
                _names[fullname] = author_id
                _next_id += 1

            # append author id to paper's list; set PACS the first time
            if article_id in _papers:
                _papers[article_id]["authors"].append(author_id)
            else:
                _papers[article_id] = {"authors": [author_id], "PACS": pacs}

            count += 1
            # if (count & 0x3FFF) == 0:  # every 16384 rows
            #     print(f"rows processed: {count}")

    # sync next id back (if you need it)
    # next_name_id = _next_id
    return _papers


def load_PACS(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    papers = load_PACS_common()
    hg = Hypergraph()
    for id, paper in papers.items():
        autohrs = paper["authors"]
        if len(autohrs) > 1 and (
            construction_method.limit_edge_size == None or len(autohrs) <= construction_method.limit_edge_size
        ):
            if not hg.has_edge_with_nodes(autohrs):
                if construction_method.weighted:
                    hg.add_edge(Hyperedge(autohrs, 0.0))
                else:
                    hg.add_edge(Hyperedge(autohrs))

            if construction_method.weighted:
                handle = hg.get_first_edges_by_nodes(autohrs)
                handle.weight += 1.0

    return hg


def pickle_PACS():
    import pandas as pd

    tb = pd.read_csv("{}/PACS.csv".format(cfg.DATASET_DIR))

    tb = tb[["ArticleID", "PACS", "FullName"]]

    papers = {}

    c = 0

    names = {}
    nidx = 0

    for _, row in tb.iterrows():
        idx = str(row["ArticleID"])
        a = str(row["PACS"])
        b = str(row["FullName"])

        if b in names:
            b = names[b]
        else:
            names[b] = nidx
            nidx += 1
            b = names[b]

        if idx in papers:
            papers[idx]["authors"].append(b)
        else:
            papers[idx] = {}
            papers[idx]["authors"] = [b]
            papers[idx]["PACS"] = a

        c += 1
        if c % 1000 == 0:
            print(c, tb.shape)

    import pickle

    pickle.dump(papers, open("PACS.pickle", "wb"))

    # for k in papers:
    #    print(papers[k])


def load_PACS_pickled(N):
    import pickle

    papers = pickle.load(open("PACS.pickle", "rb"))

    edges = []

    tot = []

    for k in papers:
        authors = papers[k]["authors"]
        if len(authors) > 1 and len(authors) <= N:
            edges.append(tuple(sorted(authors)))
        tot.append(tuple(sorted(authors)))

    # plot_dist_hyperedges(tot, "PACS")
    # print(len(edges))
    return edges


def load_PACS_single_pickled(N, S):
    import pickle

    papers = pickle.load(open("PACS.pickle", "rb"))

    edges = []

    tot = []

    for k in papers:
        if int(papers[k]["PACS"]) != S:
            continue
        authors = papers[k]["authors"]
        if len(authors) > 1 and len(authors) <= N:
            edges.append(tuple(sorted(authors)))
        tot.append(tuple(sorted(authors)))

    ##plot_dist_hyperedges(tot, "PACS")
    # print(count(tot))
    # print(len(edges))
    return edges


def load_workspace(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    import networkx as nx

    dataset = "{}/workspace.dat".format(cfg.DATASET_DIR)

    fopen = open(dataset, "r")
    lines = fopen.readlines()

    graph = {}
    for l in lines:
        t, a, b = l.split()
        t = int(t) - 28820
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]

    fopen.close()

    def standard_construction():
        cm = cast(StandardConstructionMethod, construction_method)
        hg = Hypergraph()

        for k in graph.keys():
            e_k = graph[k]
            G = nx.Graph(e_k, directed=False)
            c = list(nx.find_cliques(G))
            for i in c:
                i = tuple(sorted(i))

                if not (cm.limit_edge_size and len(i) > cm.limit_edge_size):
                    if cm.weighted:
                        if hg.has_edge_with_nodes(i):
                            handle = hg.get_first_edges_by_nodes(i)
                            handle.weight += 1.0
                        else:
                            hg.add_edge(Hyperedge(i, 1.0))
                    else:
                        if not hg.has_edge_with_nodes(i):
                            hg.add_edge(Hyperedge(i))

        hg.normalize_weights(cm.normalization_method)
        return hg

    def time_window_construction():
        cm = cast(TimeWindowConstructionMethod, construction_method)
        raise NotImplementedError()

    def temporal_path_construction():
        cm = cast(TemporalPathConstructionMethod, construction_method)
        raise NotImplementedError()

    if isinstance(construction_method, StandardConstructionMethod):
        return standard_construction()
    elif isinstance(construction_method, TimeWindowConstructionMethod):
        return time_window_construction()
    elif isinstance(construction_method, TemporalPathConstructionMethod):
        return temporal_path_construction()
    else:
        raise ValueError("Unknown construction method")


def load_DBLP(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    # dataset = "{}/dblp.csv".format(cfg.DATASET_DIR)
    # fopen = open(dataset, "r")
    # lines = fopen.readlines()
    graph = {}

    with open("{}/dblp.csv".format(cfg.DATASET_DIR), newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader, None)
        count = 0
        for row in reader:
            paper, author, y = row
            y = int(y)
            if paper not in graph:
                graph[paper] = []
            graph[paper].append(author)

    def standard_construction():
        cm = cast(StandardConstructionMethod, construction_method)
        hg = Hypergraph()

        for paper, authors in graph.items():
            if not (cm.limit_edge_size and len(authors) > cm.limit_edge_size):
                if cm.weighted:
                    if hg.has_edge_with_nodes(authors):
                        handle = hg.get_first_edges_by_nodes(authors)
                        handle.weight += 1.0
                    else:
                        hg.add_edge(Hyperedge(authors, 1.0))
                else:
                    if not hg.has_edge_with_nodes(authors):
                        hg.add_edge(Hyperedge(authors))

        hg.normalize_weights(cm.normalization_method)
        return hg

    def time_window_construction():
        cm = cast(TimeWindowConstructionMethod, construction_method)
        raise NotImplementedError()

    def temporal_path_construction():
        cm = cast(TemporalPathConstructionMethod, construction_method)
        raise NotImplementedError()

    if isinstance(construction_method, StandardConstructionMethod):
        return standard_construction()
    elif isinstance(construction_method, TimeWindowConstructionMethod):
        return time_window_construction()
    elif isinstance(construction_method, TemporalPathConstructionMethod):
        return temporal_path_construction()
    else:
        raise ValueError("Unknown construction method")


def load_history(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    dataset = f"{cfg.DATASET_DIR}/history.csv"
    chunksize = 100_000
    lines_count = 2_367_290
    data = pd.read_csv(dataset, delimiter=",", chunksize=chunksize)
    graph = defaultdict(list)

    for chunk in track(data, total=lines_count / chunksize, description="Reading csv file"):
        for row in chunk.itertuples(index=False):
            paper, author, y = row
            # y = int(y)
            graph[paper].append(author)

    def standard_construction():
        cm = cast(StandardConstructionMethod, construction_method)
        hg = Hypergraph()

        for paper, authors in track(graph.items(), description="Constructing hypergraph"):
            if not (cm.limit_edge_size and len(authors) > cm.limit_edge_size):
                if cm.weighted:
                    if hg.has_edge_with_nodes(authors):
                        handle = hg.get_first_edges_by_nodes(authors)
                        handle.weight += 1.0
                    else:
                        hg.add_edge(Hyperedge(authors, 1.0))
                else:
                    if not hg.has_edge_with_nodes(authors):
                        hg.add_edge(Hyperedge(authors))

        hg.normalize_weights(cm.normalization_method)
        return hg

    def time_window_construction():
        cm = cast(TimeWindowConstructionMethod, construction_method)
        raise NotImplementedError()

    def temporal_path_construction():
        cm = cast(TemporalPathConstructionMethod, construction_method)
        raise NotImplementedError()

    if isinstance(construction_method, StandardConstructionMethod):
        return standard_construction()
    elif isinstance(construction_method, TimeWindowConstructionMethod):
        return time_window_construction()
    elif isinstance(construction_method, TemporalPathConstructionMethod):
        return temporal_path_construction()
    else:
        raise ValueError("Unknown construction method")


def load_geology(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    dataset = f"{cfg.DATASET_DIR}/geology.csv"
    chunksize = 100_000
    lines_count = 4_418_885
    data = pd.read_csv(dataset, delimiter=",", chunksize=chunksize)
    graph = defaultdict(list)

    for chunk in track(data, total=lines_count / chunksize, description="Reading csv file"):
        for row in chunk.itertuples(index=False):
            paper, author, y = row
            # y = int(y)
            graph[paper].append(author)

    def standard_construction():
        cm = cast(StandardConstructionMethod, construction_method)
        hg = Hypergraph()

        for paper, authors in track(graph.items(), description="Constructing hypergraph"):
            if not (cm.limit_edge_size and len(authors) > cm.limit_edge_size):
                if cm.weighted:
                    if hg.has_edge_with_nodes(authors):
                        handle = hg.get_first_edges_by_nodes(authors)
                        handle.weight += 1.0
                    else:
                        hg.add_edge(Hyperedge(authors, 1.0))
                else:
                    if not hg.has_edge_with_nodes(authors):
                        hg.add_edge(Hyperedge(authors))

        hg.normalize_weights(cm.normalization_method)
        return hg

    def time_window_construction():
        cm = cast(TimeWindowConstructionMethod, construction_method)
        raise NotImplementedError()

    def temporal_path_construction():
        cm = cast(TemporalPathConstructionMethod, construction_method)
        raise NotImplementedError()

    if isinstance(construction_method, StandardConstructionMethod):
        return standard_construction()
    elif isinstance(construction_method, TimeWindowConstructionMethod):
        return time_window_construction()
    elif isinstance(construction_method, TemporalPathConstructionMethod):
        return temporal_path_construction()
    else:
        raise ValueError("Unknown construction method")


def load_justice_ideo(
    construction_method: ConstructionMethodBase = StandardConstructionMethod(),
) -> tuple[Hypergraph, dict[int, float]]:
    cm = cast(StandardConstructionMethod, construction_method)

    # cases dataset
    dataset = "{}/justice.csv".format(cfg.DATASET_DIR)
    chunksize = 100
    lines_count = 80_846
    data = pd.read_csv(dataset, delimiter=",", chunksize=chunksize, encoding="latin-1")

    case_id = 0
    justice_name = 54
    vote = 55

    cases = {}
    nodes = {}
    idx = 0

    for chunk in track(data, total=lines_count / chunksize, description="Reading justice csv file"):
        for row in chunk.itertuples(index=False):
            c, n, v = row[case_id], row[justice_name], row[vote]
            # print(c, n, v)

            try:
                v = int(v)  # valid vote
            except:
                continue  # not voted

            if n in nodes:
                n = nodes[n]
            else:
                nodes[n] = idx
                idx += 1
                n = nodes[n]

            if c in cases:
                if v in cases[c]:
                    cases[c][v].append(n)
                else:
                    cases[c][v] = [n]
            else:
                cases[c] = {}
                cases[c][v] = [n]

    # ideology dataset
    ideo = "{}/justices_ideology.csv".format(cfg.DATASET_DIR)
    chunksize = 100
    lines_count = 177
    ideo_data = pd.read_csv(ideo, delimiter=",", chunksize=chunksize, encoding="latin-1")

    spaethid = 8
    ideo = 214

    dict_ideo = {}
    for chunk in track(ideo_data, total=lines_count / chunksize, description="Reading justice ideology csv file"):
        for row in chunk.itertuples(index=False):
            ID, v = row[spaethid], row[ideo]
            try:
                ID = int(ID)
                v = float(v)
            except:
                continue
            dict_ideo[ID] = v

    hg = Hypergraph()
    for k in cases:
        for v in cases[k]:
            e = cases[k][v]
            if len(e) > 1 and not (cm.limit_edge_size and len(e) > cm.limit_edge_size):
                if cm.weighted:
                    if hg.has_edge_with_nodes(e):
                        handle = hg.get_first_edges_by_nodes(e)
                        handle.weight += 1.0
                    else:
                        hg.add_edge(Hyperedge(e, 1.0))
                else:
                    if not hg.has_edge_with_nodes(e):
                        hg.add_edge(Hyperedge(e))

    # plot_dist_hyperedges(tot, "justice")
    # print(len(edges))
    return hg, dict_ideo


def load_justice(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    cm = cast(StandardConstructionMethod, construction_method)
    dataset = "{}/justice.csv".format(cfg.DATASET_DIR)
    chunksize = 100
    lines_count = 80_846

    data = pd.read_csv(dataset, delimiter=",", chunksize=chunksize, encoding="latin-1")

    cases = {}
    nodes = {}
    idx = 0

    case_id = 0
    justice_name = 54
    vote = 55

    for chunk in track(data, total=lines_count / chunksize, description="Reading csv file"):
        for row in chunk.itertuples(index=False):
            c, n, v = row[case_id], row[justice_name], row[vote]
            # print(c, n, v)

            try:
                v = int(v)  # valid vote
            except:
                continue  # not voted

            if n in nodes:
                n = nodes[n]
            else:
                nodes[n] = idx
                idx += 1
                n = nodes[n]

            if c in cases:
                if v in cases[c]:
                    cases[c][v].append(n)
                else:
                    cases[c][v] = [n]
            else:
                cases[c] = {}
                cases[c][v] = [n]

    hg = Hypergraph()
    for k in cases:
        for v in cases[k]:
            e = cases[k][v]
            if len(e) > 1 and not (cm.limit_edge_size and len(e) > cm.limit_edge_size):
                if cm.weighted:
                    if hg.has_edge_with_nodes(e):
                        handle = hg.get_first_edges_by_nodes(e)
                        handle.weight += 1.0
                    else:
                        hg.add_edge(Hyperedge(e, 1.0))
                else:
                    if not hg.has_edge_with_nodes(e):
                        hg.add_edge(Hyperedge(e))

    # plot_dist_hyperedges(tot, "justice")
    # print(len(edges))
    # print(count(tot))
    return hg


def load_babbuini(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    import gzip

    import networkx as nx

    f = gzip.open("{}/babbuini.txt".format(cfg.DATASET_DIR), "rb")
    lines = f.readlines()

    graph = {}
    names = {}
    idx = 0

    cont = 0
    for l in lines:
        # print(l)
        if cont == 0:
            cont = 1
            continue

        l = l.split()

        t, a, b, _, _ = l

        t = int(t)

        if a in names:
            a = names[a]
        else:
            names[a] = idx
            a = idx
            idx += 1

        if b in names:
            b = names[b]
        else:
            names[b] = idx
            b = idx
            idx += 1

        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]

    def standard_construction():
        cm = cast(StandardConstructionMethod, construction_method)
        hg = Hypergraph()

        for k in graph.keys():
            e_k = graph[k]
            G = nx.Graph(e_k, directed=False)
            c = list(nx.find_cliques(G))
            for e in c:
                if not (cm.limit_edge_size and len(e) > cm.limit_edge_size):
                    if cm.weighted:
                        if hg.has_edge_with_nodes(e):
                            handle = hg.get_first_edges_by_nodes(e)
                            handle.weight += 1.0
                        else:
                            hg.add_edge(Hyperedge(e, 1.0))
                    else:
                        if not hg.has_edge_with_nodes(e):
                            hg.add_edge(Hyperedge(e))

        hg.normalize_weights(cm.normalization_method)
        return hg

    def time_window_construction():
        cm = cast(TimeWindowConstructionMethod, construction_method)
        raise NotImplementedError()

    def temporal_path_construction():
        cm = cast(TemporalPathConstructionMethod, construction_method)
        raise NotImplementedError()

    if isinstance(construction_method, StandardConstructionMethod):
        return standard_construction()
    elif isinstance(construction_method, TimeWindowConstructionMethod):
        return time_window_construction()
    elif isinstance(construction_method, TemporalPathConstructionMethod):
        return temporal_path_construction()
    else:
        raise ValueError("Unknown construction method")


def load_wiki(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    if not isinstance(construction_method, StandardConstructionMethod):
        raise ValueError("Unsupported construction method")

    fopen = open("{}/wiki.txt".format(cfg.DATASET_DIR), "r")
    lines = fopen.readlines()

    edges = set()
    tot = set()
    votes = {}

    hg = Hypergraph()
    for l in lines:
        l = l.split()

        if len(l) == 0:
            for k in votes:
                cm = construction_method
                e = tuple(sorted(votes[k]))
                tot.add(e)

                if not (cm.limit_edge_size and len(e) > cm.limit_edge_size):
                    if cm.weighted:
                        if hg.has_edge_with_nodes(e):
                            handle = hg.get_first_edges_by_nodes(e)
                            handle.weight += 1.0
                        else:
                            hg.add_edge(Hyperedge(e, 1.0))
                    else:
                        if not hg.has_edge_with_nodes(e):
                            hg.add_edge(Hyperedge(e))

            votes = {}
            continue

        if l[0] != "V":
            continue

        _, vote, u_id, _, _, _ = l
        if vote in votes:
            votes[vote].append(u_id)
        else:
            votes[vote] = [u_id]

    # console.print("[yellow][Warning] Ignoring construction method; defaulting to 2-edges only [/]")
    ##plot_dist_hyperedges(tot, "wiki")
    # print(len(edges))
    # print(count(tot))
    return hg


def load_NDC_substances(construction_method: ConstructionMethodBase = StandardConstructionMethod()) -> Hypergraph:
    if not isinstance(construction_method, StandardConstructionMethod):
        raise ValueError("Unsupported construction method")

    cm = cast(StandardConstructionMethod, construction_method)
    p = "{}/NDC-substances/".format(cfg.DATASET_DIR)
    a = open(p + "NDC-substances-nverts.txt")
    b = open(p + "NDC-substances-simplices.txt")
    v = list(map(int, a.readlines()))
    s = list(map(int, b.readlines()))
    a.close()
    b.close()

    hg = Hypergraph()
    for i in v:
        cont = 0
        e = []
        while cont < i:
            e.append(s.pop(0))
            cont += 1

        if not (cm.limit_edge_size and len(e) > cm.limit_edge_size):
            if cm.weighted:
                if hg.has_edge_with_nodes(e):
                    handle = hg.get_first_edges_by_nodes(e)
                    handle.weight += 1.0
                else:
                    hg.add_edge(Hyperedge(e, 1.0))
            else:
                if not hg.has_edge_with_nodes(e):
                    hg.add_edge(Hyperedge(e))

    return hg


def load_NDC_classes(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    if not isinstance(construction_method, StandardConstructionMethod):
        raise ValueError("Unsupported construction method")

    cm = cast(StandardConstructionMethod, construction_method)
    p = "{}/NDC-classes/".format(cfg.DATASET_DIR)
    a = open(p + "NDC-classes-nverts.txt")
    b = open(p + "NDC-classes-simplices.txt")
    v = list(map(int, a.readlines()))
    s = list(map(int, b.readlines()))
    a.close()
    b.close()

    hg = Hypergraph()
    for i in v:
        cont = 0
        e = []
        while cont < i:
            e.append(s.pop(0))
            cont += 1
        if not (cm.limit_edge_size and len(e) > cm.limit_edge_size):
            if cm.weighted:
                if hg.has_edge_with_nodes(e):
                    handle = hg.get_first_edges_by_nodes(e)
                    handle.weight += 1.0
                else:
                    hg.add_edge(Hyperedge(e, 1.0))
            else:
                if not hg.has_edge_with_nodes(e):
                    hg.add_edge(Hyperedge(e))

    return hg


def load_eu(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    if not isinstance(construction_method, StandardConstructionMethod):
        raise ValueError("Unsupported construction method")

    cm = cast(StandardConstructionMethod, construction_method)
    name = "email-Eu"
    p = f"{cfg.DATASET_DIR}/{name}/"
    a = open(p + "{}-nverts.txt".format(name))
    b = open(p + "{}-simplices.txt".format(name))
    v = list(map(int, a.readlines()))
    s = list(map(int, b.readlines()))
    a.close()
    b.close()

    hg = Hypergraph()
    s_idx = 0
    for i in track(v, description="Constructing hypergraph"):
        cont = 0
        e = []
        while cont < i:
            e.append(s[s_idx])
            s_idx += 1
            cont += 1
        if not (cm.limit_edge_size and len(e) > cm.limit_edge_size):
            if cm.weighted:
                if hg.has_edge_with_nodes(e):
                    handle = hg.get_first_edges_by_nodes(e)
                    handle.weight += 1.0
                else:
                    hg.add_edge(Hyperedge(e, 1.0))
            else:
                if not hg.has_edge_with_nodes(e):
                    hg.add_edge(Hyperedge(e))

    return hg


def load_enron(construction_method: ConstructionMethodBase = StandardConstructionMethod()):
    if not isinstance(construction_method, StandardConstructionMethod):
        raise ValueError("Unsupported construction method")

    cm = cast(StandardConstructionMethod, construction_method)
    name = "email-Enron"
    p = f"{cfg.DATASET_DIR}/{name}/"
    a = open(p + "{}-nverts.txt".format(name))
    b = open(p + "{}-simplices.txt".format(name))
    v = list(map(int, a.readlines()))
    s = list(map(int, b.readlines()))
    a.close()
    b.close()

    hg = Hypergraph()

    for i in v:
        cont = 0
        e = []
        while cont < i:
            e.append(s.pop(0))
            cont += 1
        if not (cm.limit_edge_size and len(e) > cm.limit_edge_size):
            if cm.weighted:
                if hg.has_edge_with_nodes(e):
                    handle = hg.get_first_edges_by_nodes(e)
                    handle.weight += 1.0
                else:
                    hg.add_edge(Hyperedge(e, 1.0))
            else:
                if not hg.has_edge_with_nodes(e):
                    hg.add_edge(Hyperedge(e))

    return hg


# Example usage
# ConstructionMethod.standard(False, NormalizationMethod.DEFAULT)
# ConstructionMethod.time_window(10.0, True, NormalizationMethod.RANKING)
