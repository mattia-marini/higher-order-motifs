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
