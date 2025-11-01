from logging import currentframe
import random, math
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from networkx import graph
import numpy as np
import os
import config as cfg
import hypergraphx as hx
from hypergraphx.viz import draw_hypergraph

def motifs_ho_not_full(edges, N, TOT, visited):
    """ Computes the motif count for hypergraph motifs of order N. The
    subgraphs checked for motifs are the ones obtained by extending a hyperedge
    of order N-1 with one of its neighboring hyperedges. The subgraphs
    contained in the visited set are ignored

    Args:
        edges (list[tuple[int]]): List of hyperedges in the hypergraph.
        N (int): The order of the motifs to be counted.
        TOT (int): ??

    Returns: 
        out (list[tuple[tuple[tuple[int]], int]]): A list of tuples where each
            tuple contains a motif (as a tuple of edges) and its corresponding
            count in the hypergraph.
        visited (dict[tuple[int], int]): A dictionary of visited hyperedges of
            size N.
        
    """
    mapping, labeling = generate_motifs(N)

    T = {}
    graph = {}
    for e in edges:
        if len(e) >= N:
            continue

        T[tuple(sorted(e))] = 1

        for e_i in e:
            if e_i in graph:
                graph[e_i].append(e)
            else:
                graph[e_i] = [e]

    def count_motif(nodes):
        nodes = tuple(sorted(tuple(nodes)))
        p_nodes = power_set(nodes)
        
        motif = []
        for edge in p_nodes:
            if len(edge) >= 2:
                edge = tuple(sorted(list(edge)))
                if edge in T:
                    motif.append(edge)

        m = {}
        idx = 1
        for i in nodes:
            m[i] = idx
            idx += 1

        labeled_motif = []
        for e in motif:
            new_e = []
            for node in e:
                new_e.append(m[node])
            new_e = tuple(sorted(new_e))
            labeled_motif.append(new_e)
        labeled_motif = tuple(sorted(labeled_motif))

        if labeled_motif in labeling:
            labeling[labeled_motif] += 1

    for e in edges:
        if len(e) == N - 1:
            nodes = list(e)
            
            for n in nodes:
                for e_i in graph[n]:
                    tmp = list(nodes)
                    tmp.extend(e_i)
                    tmp = list(set(tmp))
                    if len(tmp) == N and not (tuple(sorted(tmp)) in visited):
                        visited[tuple(sorted(tmp))] = 1
                        count_motif(tmp)

    out = []

    for motif in mapping.keys():
        count = 0
        for label in mapping[motif]:
            count += labeling[label]
            
        out.append((motif, count))

    out = list(sorted(out))

    D = {}
    for i in range(len(out)):
        D[i] = out[i][0]
    
    #with open('motifs_{}.pickle'.format(N), 'wb') as handle:
        #pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return out, visited

def motifs_standard(edges, N, TOT, visited):
    """
    Computes the motif count for hypergraph motifs of order N, considering only
    hyperedges of order 2. The subgraphs contained in the visited set are
    ignored.

    Args:
        edges (list[tuple[int]]): List of hyperedges in the hypergraph.
        N (int): The order of the motifs to be counted.
        TOT (int): ??
        visited (set[tuple[int]]): A set of subgraph (tuple of nodes) to ignore
            in the computation of motifs

    Returns:
        out (list[tuple[tuple[tuple[int]], int]]): A list of tuples where each
            tuple contains a motif (as a tuple of edges) and its corresponding
            count in the hypergraph.
    """
    mapping, labeling = generate_motifs(N)

    graph = {}
    T = {}

    z = set()
    for e in edges:
        for n in e:
            z.add(n)

    # Construct adjacency matrix for 2-edges
    for e in edges:
        if len(e) == 2:
            T[tuple(sorted(e))] = 1
            a, b = e
            if a in graph:
                graph[a].append(b)
            else:
                graph[a] = [b]

            if b in graph:
                graph[b].append(a)
            else:
                graph[b] = [a]

    def count_motif(nodes):
        nodes = tuple(sorted(tuple(nodes)))

        if nodes in visited:
            return

        p_nodes = power_set(nodes)
        
        motif = []
        for edge in p_nodes:
            edge = tuple(sorted(list(edge)))
            if edge in T:
                motif.append(edge)

        m = {}
        idx = 1
        for i in nodes:
            m[i] = idx
            idx += 1

        labeled_motif = []
        for e in motif:
            new_e = []
            for node in e:
                new_e.append(m[node])
            new_e = tuple(sorted(new_e))
            labeled_motif.append(new_e)
        labeled_motif = tuple(sorted(labeled_motif))

        if labeled_motif in labeling:
            labeling[labeled_motif] += 1

    def graph_extend(sub, ext, v, n_sub):

        if len(sub) == N:
            count_motif(sub)
            return

        while len(ext) > 0:
            w = ext.pop()
            tmp = set(ext)

            for u in graph[w]:
                if u not in sub and u not in n_sub and u > v:
                    tmp.add(u)

            new_sub = set(sub)
            new_sub.add(w)
            new_n_sub = set(n_sub).union(set(graph[w]))
            graph_extend(new_sub, tmp, v, new_n_sub)

    c = 0
    
    k = 0
    for v in graph.keys():
        v_ext = set()
        for u in graph[v]:
            if u > v:
                v_ext.add(u)
        k += 1
        if k % 5 == 0:
            print(k, len(z), TOT)

        graph_extend(set([v]), v_ext, v, set(graph[v]))
        c += 1

    out = []

    for motif in mapping.keys():
        count = 0
        for label in mapping[motif]:
            count += labeling[label]
            
        out.append((motif, count))

    out = list(sorted(out))

    D = {}
    for i in range(len(out)):
        D[i] = out[i][0]
    
    #with open('motifs_{}.pickle'.format(N), 'wb') as handle:
        #pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return out

def motifs_ho_full(edges, N, TOT):
    """
    Computes the motif counts for hypergraph motifs of order N. The subgraphs
    checked for motifs are the one induced by the nodes in the hyperedges of
    order N

    Args:
        edges (list[tuple[int]]): List of hyperedges in the hypergraph.
        N (int): The order of the motifs to be counted.
        TOT (int): ??

    Returns: 
        out (list[tuple[tuple[tuple[int]], int]]): A list of tuples where each
            tuple contains a motif (as a tuple of edges) and its corresponding
            count in the hypergraph.
        visited (dict[tuple[int], int]): A dictionary of visited hyperedges of
            size N.
        
    """
    mapping, labeling = generate_motifs(N)

    T = {}
    for e in edges:
        T[tuple(sorted(e))] = 1

    visited = {}

    def count_motif(nodes):
        nodes = tuple(sorted(tuple(nodes)))
        p_nodes = power_set(nodes)
        
        motif = []
        # TODO make polinomial in number of verticies
        for edge in p_nodes:
            if len(edge) >= 2:
                edge = tuple(sorted(list(edge)))
                if edge in T:
                    motif.append(edge)

        m = {}
        idx = 1
        for i in nodes:
            m[i] = idx
            idx += 1

        labeled_motif = []
        for e in motif:
            new_e = []
            for node in e:
                new_e.append(m[node])
            new_e = tuple(sorted(new_e))
            labeled_motif.append(new_e)
        labeled_motif = tuple(sorted(labeled_motif))

        if labeled_motif in labeling:
            labeling[labeled_motif] += 1

    for e in edges:
        if len(e) == N:
            #print(e)
            visited[e] = 1
            nodes = list(e)
            count_motif(nodes)

    out = []

    for motif in mapping.keys():
        count = 0
        for label in mapping[motif]:
            count += labeling[label]
            
        out.append((motif, count))

    out = list(sorted(out))

    D = {}
    for i in range(len(out)):
        D[i] = out[i][0]
    
    #with open('motifs_{}.pickle'.format(N), 'wb') as handle:
        #pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return out, visited

def diff_sum(original, null_models):
    u_null = avg(null_models)

    res = []
    for i in range(len(original)):
        res.append((original[i][1] - u_null[i]) / (original[i][1] + u_null[i] + 4))

    return res

def norm_vector(a):
    res = []
    M = 0
    for i in a:
        M += i**2
    M = math.sqrt(M)
    res = [i / M for i in a]
    return res

def count(edges):
    """
    Prints information about the hypergraph: 
    - number of nodes
    - number of hyperedges
    - count of hyperedges of sizes 2 to 5.
    """
    d = {}
    n = set()
    for i in range(2, 6):
        d[i] = 0
    for i in edges:
        try:
            d[len(i)] += 1
        except:
            pass
        finally:
            for j in i:
                n.add(j)
    print("& {} & {} & {} & {} & {} & {}".format(len(n), len(edges), d[2], d[3], d[4], d[5]))

def count_weight(edges):
    """
    Prints information about the hypergraph: 
    - number of nodes
    - number of hyperedges
    - count of hyperedges of sizes 2 to 5 and the average weight
    """
    d = {}
    w = {}
    n = set()
    for i in range(2, 6):
        d[i] = 0
        w[i] = 0
    for edge, weight in edges.items():
        try:
            d[len(edge)] += 1
            w[len(edge)] += weight
        except:
            pass
        finally:
            for j in edge:
                n.add(j)

    for i in range(2, 6):
        if d[i] == 0:
            w[i] = 0
        else:
            w[i] = w[i]/d[i]
    print("& {} & {} & {}:{:.3f} & {}:{:.3f} & {}:{:.3f} & {}:{:.3f}".format(len(n), len(edges), d[2], w[2], d[3], w[3], d[4],w[4], d[5], w[5]))

def plot_dist_hyperedges(edges, title):
    os.makedirs(cfg.PLOT_OUT_DIR, exist_ok=True)

    x = []
    for i in edges:
        if len(i) > 1 and len(i) < 20:
            x.append(len(i))

    M = max(x)

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(title)
    plt.xlabel("Size S")
    plt.ylabel("P(S)")

    ax.hist(x, bins=np.arange(M + 1)+0.5, density=True, alpha=0.5, histtype='bar', ec='black')
    plt.savefig("{}/{}.pdf".format(cfg.PLOT_OUT_DIR, title))

def plot_dist_hyperedges_weights(edges, title, min_size=2, max_size=20):
    """
    edges: dict[tuple[int], int] -- key is hyperedge (tuple of vertices), value is its weight
    """
    # 1. Group weights by edge size
    size2weights = {}
    for edge, weight in edges.items():
        size = len(edge)
        if min_size <= size <= max_size:
            size2weights.setdefault(size, []).append(weight)

    # 2. Plot distributions
    os.makedirs(cfg.PLOT_OUT_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(2, 1, figsize=(5, 10),gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.6})
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_xlabel("w")
    ax[0].set_ylabel("count(w)")

    ax[0].set_title(f"{title}\n")
    ax[0].text(
        0.5, 1.01,
        "The number of weights by hyperedge size",
        ha='center', va='bottom',
        transform=ax[0].transAxes,
        fontsize=8, fontstyle='italic'
    )
    ax[0].text(0.5, -0.2, "The graph purposely keeps only the percentile range containing 80% of the weights,\n centered around the expected value, to exclude outliers.", 
           ha='center', va='center', transform=ax[0].transAxes, fontsize=8)


    # Calculate the percentile range containing 80% of the weights.
    total_weight_count = np.zeros(max(edges.values()) + 1)
    for w in edges.values():
        total_weight_count[w] += 1

    weight_dist = total_weight_count/len(edges)

    # Exclude outliers by focusing on central 80% around expected value
    expected_value = 0
    for w, p in enumerate(weight_dist):
        expected_value += w * p

    target_count = round(0.8 * len(edges)) # I want to include at least 80% of edges
    current_count = total_weight_count[round(expected_value)] # Start from expected value
    l = round(expected_value) - 1
    r = round(expected_value) + 1

    while current_count < target_count:
        if l >= 0:
            current_count += total_weight_count[l]
            l -= 1
        if r < len(total_weight_count):
            current_count += total_weight_count[r]
            r += 1
    l=max(l,0)


    max_hyperedge_size = max(size2weights.keys()) + 1
    discarded_values = [list() for _ in range(max_hyperedge_size)]

    for idx, (size, weights) in enumerate(sorted(size2weights.items())):
        if len(weights) == 0:
            continue

        curr_weight_count = np.zeros(r + 1)
        for w in weights:
            try:
                curr_weight_count[w] += 1
            except:
                discarded_values[size].append(w)

        # print(f"{size} - |e|:{len(weights)} e: {expected_value} l:{l} r:{r}")
        ax[0].plot(
            range(l,r),
            curr_weight_count[l:r],
            label=f"size={size}",
            alpha=0.8,
        )

    ax[0].legend(title="Hyperedge size")

    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[1].set_xlabel("hyperedge size S")
    ax[1].set_ylabel("count(outliers)")

    ax[1].set_title(f"Outliers discarded from {title}\n")
    ax[1].text(
        0.5, 1.01,
        "The number of edges discarded as outliers by hyperedge size",
        ha='center', va='bottom',
        transform=ax[1].transAxes,
        fontsize=8, fontstyle='italic'
    )

    discarded_values_count = [len(dv) for dv in discarded_values]

    ax[1].bar(
        range(2,max_hyperedge_size),
        discarded_values_count[2:],
        alpha=0.5,
    )
    

    fig.savefig(f"{cfg.PLOT_OUT_DIR}/{title}_weights_by_size.pdf", bbox_inches="tight")
    plt.close()

def plot_dist_motifs(motifs, title, graphs_per_row = None):
    os.makedirs(cfg.PLOT_OUT_DIR, exist_ok=True)
    n = len(motifs)

    if graphs_per_row is None:
        graphs_per_row = n


    counts = [c for (_, c) in motifs]
    motifs_reprentations = [m for (m, _) in motifs]
    m = math.ceil(n / graphs_per_row)

    fig = plt.figure(figsize=(5, 5 + 5/graphs_per_row * m))
    gs = gridspec.GridSpec(1 + m, graphs_per_row, height_ratios=[graphs_per_row] + [1] * m,  wspace=0)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(title)
    plt.xlabel("Motif id")
    plt.ylabel("Count")

    ax0.bar(range(len(counts)), counts, align='center', alpha=0.7)

    for i, motif in enumerate(motifs_reprentations):
        # if i == 10: 
        #     break

        nodes = set()
        for e in motif:
            for v in e:
                nodes.add(v)

        node_size=150
        hyperedge_alpha=0.8
        pos = {1:(-1,0), 2:(1,0), 3:(0,1.5)}

        if len(nodes) == 4:
            node_size = 50
            hyperedge_alpha=0.3
            pos = {1:(-1,-1), 2:(1,-1), 3:(1,1), 4:(-1, 1)}

        # print(f"Plotting motif {i}")
        ax = fig.add_subplot(gs[1 + math.floor(i/graphs_per_row), i % graphs_per_row])
        ax.set_frame_on(False)
        ax.set_xlabel(str(i), fontsize=8)
        # ax.axis('off')

        
        hypergraph = hx.Hypergraph()
        hypergraph.add_edges(list(motif))
        draw_hypergraph(hypergraph, ax=ax, edge_color="black", pos=pos, node_size=node_size, hyperedge_alpha=hyperedge_alpha)
        

    plt.tight_layout()

    fig.savefig("{}/{}.pdf".format(cfg.PLOT_OUT_DIR, title))

def avg(motifs):
    result = []
    for i in range(len(motifs[0])):
        s = 0
        for j in range(len(motifs)):
            s += motifs[j][i][1]

        result.append(s / len(motifs))
    return result

def sigma(motifs):
    u = avg(motifs)

    result = []
    for i in range(len(motifs[0])):
        s = 0
        for j in range(len(motifs)):
            s += (motifs[j][i][1] - u[i])**2
        s /= len(motifs)
        s = s ** 0.5

        result.append(s)
    return result

def z_score(original, null_models):
    u_null = avg(null_models)
    sigma_null = sigma(null_models)

    z_scores = []
    for i in range(len(original)):
        z_scores.append((original[i][1] - u_null[i]) / (sigma_null[i] + 0.01))

    return z_scores

def power_set(A): 
    """
    Generate the power set of a given set A.

    Args:
        A (list): The input set.

    Returns:
        list: A list containing all subsets of A.
    """

    subsets = []
    N = len(A)

    for mask in range(1<<N):
        subset = []

        for n in range(N):
            if ((mask>>n)&1) == 1:
                subset.append(A[n])

        subsets.append(subset)

    return subsets

def is_connected(edges, N):
    nodes = set()
    for e in edges:
        for n in e:
            nodes.add(n)

    if len(nodes) != N:
        return False

    visited = {}
    for i in nodes:
        visited[i] = False
    graph = {}
    for i in nodes:
        graph[i] = []
    
    for edge in edges:
        for i in range(len(edge)):
            for j in range(len(edge)):
                if edge[i] != edge[j]:
                    graph[edge[i]].append(edge[j])
                    graph[edge[j]].append(edge[i])
    
    q = []
    nodes = list(nodes)
    q.append(nodes[0])
    while len(q) != 0:
        v = q.pop(len(q) - 1)
        if not visited[v]:
            visited[v] = True
            for i in graph[v]:
                q.append(i)
    conn = True
    for i in nodes:
        if not visited[i]:
            conn = False
            break
    return conn

def relabel(edges, relabeling):
    """
    Relabel the vertices of the edges according to the given relabeling.

    Args:
        edges (list[tuple[int]]): The list of edges to be relabeled.
        relabeling (tuple[int]): A tuple representing the new labels for the vertices.

    Returns:
        list[tuple[int]]: The relabeled edges, sorted.
    """
    res = []
    for edge in edges:
        new_edge = []
        for v in edge:
            new_edge.append(relabeling[v - 1])
        res.append(tuple(sorted(new_edge)))
    return sorted(res)

def generate_motifs(N):
    """
    Generate all isomorphism classes of connected motifs of size N.

    Args:
        N (int): The size of the motifs (number of nodes).

    Returns:
        mapping (dict[tuple[tuple[int]]: set[tuple[tuple[int]]]]): 
            A dictionary mapping each isomorphism class representative 
            to the set of all its possible labelings 
            (as tuples of edges).
        labeling (dict[tuple[tuple[int]], int]): 
            A dictionary where each key is a possible labeling of motifs of size N,
            and each value is initialized to 0.
    """
    n = N
    assert n >= 2

    h = [i for i in range(1, n + 1)]
    A = []

    for r in range(n, 1, -1):
        A.extend(list(itertools.combinations(h, r)))

    B = power_set(A)

    C = []
    for i in range(len(B)):
        if is_connected(B[i], N):
            C.append(B[i])

    isom_classes = {}

    for i in C:
        edges = sorted(i)
        relabeling_list = list(itertools.permutations([j for j in range(1, n + 1)]))
        found = False
        for relabeling in relabeling_list:
            relabeling_i = relabel(edges, relabeling)
            #print(relabeling_i)
            if tuple(relabeling_i) in isom_classes:
                found = True
                break
        if not found:
            isom_classes[tuple(edges)] = 1

    mapping = {}
    labeling = {}

    for k in isom_classes.keys():
        mapping[k] = set()
        relabeling_list = list(itertools.permutations([j for j in range(1, n + 1)]))
        for relabeling in relabeling_list:
            relabeling_i = relabel(k, relabeling)
            labeling[tuple(sorted(relabeling_i))] = 0
            mapping[k].add(tuple(sorted(relabeling_i)))
    
    return mapping, labeling

def intensity(edges):
    total = 0
    for e, w in edges.items():
        total *= w
    return total

def coherence(edges):
    mean = sum(edges.values())/len(edges)
    return intensity(edges) / mean

def induced_subgraph(edges, nodes):
    subgraph = {}
    node_set = set(nodes)
    for e, w in edges.items():
        include = True
        for n in e:
            if n not in node_set:
                include = False
                break
        if include:
            subgraph[e] = w
    return subgraph

#out = len(isom_classes.keys())
#print(out)
