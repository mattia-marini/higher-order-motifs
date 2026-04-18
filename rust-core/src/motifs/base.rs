use itertools::Itertools;
use std::collections::{HashMap, HashSet, VecDeque};

/// Node labels are expected to be 1..=n.
type Node = usize;

/// Motif generation here is *unweighted* in spirit (Python reference uses unweighted hypergraphs).
/// We nevertheless keep a "weight" field because the Python project also has weighted variants.
///
/// Important: this weight type must be `Eq + Hash + Ord` because we store hypergraphs in
/// `HashSet`/`HashMap` and compare them lexicographically for canonicalization.
type Edge = (usize, Vec<Node>); // (weight, nodes)

/// A hypergraph is a list of edges. The representation is normalized by:
/// - sorting nodes inside each edge
/// - sorting edges in the hypergraph
///
/// With this normalization, Rust's derived lexicographic ordering for `Vec` matches the
/// Python "internally and externally sorted" representation.
type Hypergraph = Vec<Edge>;

/// Relabel a hypergraph according to `mapping` and return a normalized hypergraph.
fn relabel(hg: &Hypergraph, mapping: &HashMap<usize, usize>) -> Hypergraph {
    let mut res: Hypergraph = hg
        .iter()
        .map(|(w, nodes)| {
            let mut new_nodes: Vec<Node> = nodes
                .iter()
                .map(|v| *mapping.get(v).expect("missing node in mapping"))
                .collect();
            new_nodes.sort_unstable();
            (*w, new_nodes)
        })
        .collect();

    res.sort();
    res
}

/// Powerset of the provided edge-node lists.
///
/// The input `edges` is expected to be a list of hyperedges represented by their node list.
/// Output hypergraphs are normalized.
fn power_set(edges: &Vec<Vec<Node>>) -> Vec<Hypergraph> {
    let mut edges = edges.clone();
    edges.sort();

    let m = edges.len();
    let mut subsets: Vec<Hypergraph> = Vec::with_capacity(1usize << m);

    for mask in 0usize..(1usize << m) {
        let mut g: Hypergraph = Vec::new();
        for i in 0..m {
            if ((mask >> i) & 1) == 1 {
                g.push((1, edges[i].clone()));
            }
        }
        // normalize (external)
        g.sort();
        subsets.push(g);
    }

    subsets
}

/// Check whether `g` is a connected spanning hypergraph on exactly `n` nodes.
///
/// Connectivity is computed by turning each hyperedge into a clique in the underlying
/// 2-section graph (same as the Python reference).
fn is_connected(g: &Hypergraph, n: usize) -> bool {
    if n == 0 {
        return false;
    }

    let mut nodes: HashSet<Node> = HashSet::new();
    for (_w, edge_nodes) in g.iter() {
        for &v in edge_nodes.iter() {
            nodes.insert(v);
        }
    }

    // Must span exactly all n labeled nodes.
    if nodes.len() != n {
        return false;
    }

    // adjacency list over 1..=n
    let mut adj: Vec<Vec<Node>> = vec![Vec::new(); n + 1];

    for (_w, edge_nodes) in g.iter() {
        // connect all distinct pairs inside the hyperedge
        for i in 0..edge_nodes.len() {
            for j in (i + 1)..edge_nodes.len() {
                let u = edge_nodes[i];
                let v = edge_nodes[j];
                if u == v {
                    continue;
                }
                adj[u].push(v);
                adj[v].push(u);
            }
        }
    }

    let start = *nodes.iter().next().expect("nodes non-empty if n>0");
    let mut visited = vec![false; n + 1];
    let mut q: VecDeque<Node> = VecDeque::new();
    q.push_back(start);

    while let Some(v) = q.pop_back() {
        if visited[v] {
            continue;
        }
        visited[v] = true;
        for &nei in adj[v].iter() {
            if !visited[nei] {
                q.push_back(nei);
            }
        }
    }

    // Because graph is spanning, we can just check 1..=n
    (1..=n).all(|v| visited[v])
}

/// Return every possible isomorphism (relabeling) of the given hypergraph.
///
/// If `n` is not provided, it is computed as the number of distinct nodes appearing in `hg`.
pub fn enum_isomorphisms(hg: &Hypergraph, n: Option<usize>) -> Vec<Hypergraph> {
    let n = n.unwrap_or_else(|| {
        let mut distinct: HashSet<Node> = HashSet::new();
        for (_w, edge_nodes) in hg.iter() {
            distinct.extend(edge_nodes.iter().copied());
        }
        distinct.len()
    });

    let base: Vec<Node> = (1..=n).collect();
    let mut labelings: Vec<Hypergraph> = Vec::new();

    for perm in base.iter().copied().permutations(n) {
        let mapping: HashMap<Node, Node> = (1..=n).zip(perm.into_iter()).collect();
        labelings.push(relabel(hg, &mapping));
    }

    labelings
}

/// Enumerate all connected spanning sub-hypergraphs on `n` labeled nodes.
///
/// Hyperedges are all subsets of {1..=n} of size r for r=n,n-1,...,2.
/// Sub-hypergraphs are all subsets (powerset) of those hyperedges.
pub fn enum_connected_subgraphs(n: usize) -> Vec<Hypergraph> {
    assert!(n >= 2);

    let nodes: Vec<Node> = (1..=n).collect();

    // All possible hyperedges of size 2..=n.
    let mut all_edges: Vec<Vec<Node>> = Vec::new();
    for r in (2..=n).rev() {
        all_edges.extend(nodes.iter().copied().combinations(r));
    }

    let all_subgraphs = power_set(&all_edges);

    let mut connected: Vec<Hypergraph> = Vec::new();
    for mut g in all_subgraphs {
        if is_connected(&g, n) {
            // Ensure normalized (power_set already sorts, but keep this defensive)
            for (_w, ns) in g.iter_mut() {
                ns.sort_unstable();
            }
            g.sort();
            connected.push(g);
        }
    }

    connected
}

/// Canonical representative of the isomorphism class of `hg`.
///
/// The representative is the relabeling of `hg` that is lexicographically smallest among
/// all permutations of node labels 1..=n.
pub fn get_canonical_representative(hg: &Hypergraph, n: Option<usize>) -> Hypergraph {
    let n = n.unwrap_or_else(|| {
        let mut distinct: HashSet<Node> = HashSet::new();
        for (_w, edge_nodes) in hg.iter() {
            distinct.extend(edge_nodes.iter().copied());
        }
        distinct.len()
    });

    let base: Vec<Node> = (1..=n).collect();

    let mut best: Option<Hypergraph> = None;
    for perm in base.iter().copied().permutations(n) {
        let mapping: HashMap<Node, Node> = (1..=n).zip(perm.into_iter()).collect();
        let rel = relabel(hg, &mapping);
        match &best {
            None => best = Some(rel),
            Some(b) => {
                if rel < *b {
                    best = Some(rel);
                }
            }
        }
    }

    best.expect("at least one permutation exists")
}

/// Generate all isomorphism classes of connected motifs of size `n`.
///
/// Returns:
/// - `rep_list`: sorted list of canonical representatives
/// - `rep_map`: map from every labeled motif (every isomorphism of each rep) to its rep
pub fn generate_motifs(n: usize) -> (Vec<Hypergraph>, HashMap<Hypergraph, Hypergraph>) {
    assert!(n >= 2);

    let connected_subgraphs = enum_connected_subgraphs(n);

    let mut canonical_rep: HashSet<Hypergraph> = HashSet::new();
    for g in connected_subgraphs.iter() {
        canonical_rep.insert(get_canonical_representative(g, Some(n)));
    }

    let mut rep_map: HashMap<Hypergraph, Hypergraph> = HashMap::new();
    for representative in canonical_rep.iter() {
        for iso in enum_isomorphisms(representative, Some(n)) {
            rep_map.insert(iso, representative.clone());
        }
    }

    let mut reps: Vec<Hypergraph> = canonical_rep.into_iter().collect();
    reps.sort();

    (reps, rep_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enum_connected_subgraphs_n2() {
        let graphs = enum_connected_subgraphs(2);
        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0], vec![(1, vec![1, 2])]);
    }

    #[test]
    fn test_enum_connected_subgraphs_n3_count() {
        // For n=3: 4 possible hyperedges (one 3-edge + three 2-edges) => 16 subgraphs.
        // Connected spanning ones are 12 (see reasoning in implementation notes).
        let graphs = enum_connected_subgraphs(3);
        assert_eq!(graphs.len(), 12);
        assert!(graphs.iter().all(|g| is_connected(g, 3)));
    }

    #[test]
    fn test_canonical_representative_is_in_isomorphisms() {
        let hg: Hypergraph = vec![(1, vec![1, 2]), (1, vec![2, 3])];
        let rep = get_canonical_representative(&hg, Some(3));
        let isos = enum_isomorphisms(&hg, Some(3));
        assert!(isos.contains(&rep));
    }
}
