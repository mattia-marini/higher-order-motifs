use duplicate::duplicate_item;
use foldhash::fast::FixedState;
use itertools::Itertools;
use num_traits::{AsPrimitive, PrimInt};
use pyo3::pyclass;
use seq_macro::seq;
use std::hash::Hash;
use std::{collections::VecDeque, ops::Shl};

use crate::graph::HypergraphAccessor;
use crate::graph::{hyper_adj_list::HyperAdjList, Hypergraph, NodeId};

use hashbrown::{HashMap, HashSet};

type Node = usize;
pub type UnweightedEdge = Vec<Node>;
pub type UnweightedHypergraph = Vec<UnweightedEdge>;

// Helper functions kept exactly as they are in your code for compatibility
fn relabel_unweighted(
    hg: &UnweightedHypergraph,
    mapping: &HashMap<Node, Node>,
) -> UnweightedHypergraph {
    let mut res: UnweightedHypergraph = hg
        .iter()
        .map(|nodes| {
            let mut new_nodes: Vec<Node> = nodes
                .iter()
                .map(|v| *mapping.get(v).expect("missing node in mapping"))
                .collect();
            new_nodes.sort_unstable();
            new_nodes
        })
        .collect();
    res.sort();
    res
}

pub fn enum_isomorphisms(hg: &UnweightedHypergraph, n: Option<usize>) -> Vec<UnweightedHypergraph> {
    let n = n.unwrap_or_else(|| {
        let mut distinct: HashSet<Node> = HashSet::new();
        for edge_nodes in hg.iter() {
            distinct.extend(edge_nodes.iter().copied());
        }
        distinct.len()
    });

    let base: Vec<Node> = (1..=n).collect();
    let mut labelings: Vec<UnweightedHypergraph> = Vec::new();

    for perm in base.iter().copied().permutations(n) {
        let mapping: HashMap<Node, Node> = (1..=n).zip(perm.into_iter()).collect();
        labelings.push(relabel_unweighted(hg, &mapping));
    }
    labelings
}

pub fn get_canonical_representative(
    hg: &UnweightedHypergraph,
    n: Option<usize>,
) -> UnweightedHypergraph {
    let n = n.unwrap_or_else(|| {
        let mut distinct: HashSet<Node> = HashSet::new();
        for edge_nodes in hg.iter() {
            distinct.extend(edge_nodes.iter().copied());
        }
        distinct.len()
    });

    let base: Vec<Node> = (1..=n).collect();
    let mut best: Option<UnweightedHypergraph> = None;
    for perm in base.iter().copied().permutations(n) {
        let mapping: HashMap<Node, Node> = (1..=n).zip(perm.into_iter()).collect();
        let rel = relabel_unweighted(hg, &mapping);
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

/// Fixed, blazingly fast connectivity check working directly on bitmasks
#[inline(always)]
fn is_mask_connected(mask: u32, edges: &[UnweightedEdge], n: usize) -> bool {
    // 1. Fast path: count how many nodes actually appear in this mask
    let mut node_seen = 0u8;
    let mut temp = mask;
    while temp != 0 {
        let idx = temp.trailing_zeros() as usize;
        temp &= temp - 1;
        for &node in &edges[idx] {
            node_seen |= 1 << (node - 1);
        }
    }
    // If it doesn't span exactly all n nodes, reject immediately
    if node_seen.count_ones() as usize != n {
        return false;
    }

    // 2. Compute components using standard BFS/DFS over bitset layers
    let mut adj = [0u8; 6]; // Quick adjacency bitset for up to N=5
    let mut temp = mask;
    while temp != 0 {
        let idx = temp.trailing_zeros() as usize;
        temp &= temp - 1;
        let edge = &edges[idx];
        for i in 0..edge.len() {
            for j in (i + 1)..edge.len() {
                adj[edge[i]] |= 1 << edge[j];
                adj[edge[j]] |= 1 << edge[i];
            }
        }
    }

    // Start flood fill from node 1
    let mut visited = 0u8;
    let mut q = VecDeque::with_capacity(6);
    q.push_back(1);
    visited |= 1 << 1;

    while let Some(u) = q.pop_front() {
        let mut neighbors = adj[u];
        while neighbors != 0 {
            let v = neighbors.trailing_zeros() as usize;
            neighbors &= neighbors - 1;
            if (visited & (1 << v)) == 0 {
                visited |= 1 << v;
                q.push_back(v);
            }
        }
    }

    // Check if all nodes 1..=n were visited
    let target_mask = ((1 << (n + 1)) - 1) & !1;
    (visited & target_mask) == target_mask
}

/// Decodes a flat integer bitmask back into your required UnweightedHypergraph vector structure
fn decode_mask(mut mask: u32, edges: &[UnweightedEdge]) -> UnweightedHypergraph {
    let mut hg = Vec::with_capacity(mask.count_ones() as usize);
    while mask != 0 {
        let idx = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        hg.push(edges[idx].clone());
    }
    hg.sort();
    hg
}

/// FIXED: Generates motifs using streams to prevent Out Of Memory crashes
pub fn generate_motifs(
    n: usize,
) -> (
    Vec<UnweightedHypergraph>,
    HashMap<UnweightedHypergraph, UnweightedHypergraph, FixedState>,
) {
    assert!(
        n >= 2 && n <= 5,
        "This optimized pipeline supports up to N=5"
    );

    let nodes: Vec<Node> = (1..=n).collect();
    let mut all_edges: UnweightedHypergraph = Vec::new();
    for r in (2..=n).rev() {
        all_edges.extend(nodes.iter().copied().combinations(r));
    }
    all_edges.sort();

    let total_subgraphs = 1usize << all_edges.len();
    let mut canonical_rep: HashSet<UnweightedHypergraph> = HashSet::new();

    // Loop through the powerset combinations sequentially without holding them in RAM
    for mask in 0..total_subgraphs {
        if is_mask_connected(mask as u32, &all_edges, n) {
            let g = decode_mask(mask as u32, &all_edges);
            canonical_rep.insert(get_canonical_representative(&g, Some(n)));
        }
    }

    println!("CANONICALL REPS");

    let mut rep_map: HashMap<UnweightedHypergraph, UnweightedHypergraph, FixedState> =
        HashMap::with_hasher(FixedState::default());

    for representative in canonical_rep.iter() {
        for iso in enum_isomorphisms(representative, Some(n)) {
            rep_map.insert(iso, representative.clone());
        }
    }

    let mut reps: Vec<UnweightedHypergraph> = canonical_rep.into_iter().collect();
    reps.sort();

    (reps, rep_map)
}

// #[pyclass]
#[derive(Debug, Clone)]
pub struct CanonicalRep<const N: usize, T>
where
    T: Shl + AsPrimitive<u8> + PrimInt,
{
    pub degree: [u32; N],
    pub order_map: Option<[u32; N]>,
    pub rep: Hypergraph<T, ()>,
}

impl<const N: usize, T> Hash for CanonicalRep<N, T>
where
    T: Shl + AsPrimitive<u8> + PrimInt + Hash + Eq,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.degree.hash(state);
        self.order_map.hash(state);
    }
}
impl<const N: usize, T> PartialEq for CanonicalRep<N, T>
where
    T: Shl + AsPrimitive<u8> + PrimInt + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.degree == other.degree && self.order_map == other.order_map
    }
}
impl<const N: usize, T> Eq for CanonicalRep<N, T> where T: Shl + AsPrimitive<u8> + PrimInt + Eq {}

/// for motifs of order 4, order sorting is already deterministic fingerprint
#[duplicate_item(size; [3]; [4])]
impl<T, W> From<Hypergraph<T, W>> for CanonicalRep<size, T>
where
    T: Shl + AsPrimitive<u8> + PrimInt + Hash,
{
    fn from(rep: Hypergraph<T, W>) -> Self {
        let mut degree = [0u32; size];

        seq!(M in 2..=4 {
                rep.edges::<M>().into_iter().for_each(|edge| {
                    for n in edge{
                        // let u = edge.nodes[i].as_() as usize;
                        degree[n.as_() as usize] += 1;
                    }
                });
        });

        degree.sort_unstable();

        Self {
            degree,
            order_map: None,
            rep: rep.into_unweighted(),
        }
    }
}

// impl<T, W> From<Hypergraph<T, W>> for CanonicalRep<4, T>
// where
//     T: Shl + AsPrimitive<u8> + PrimInt + Hash,
//     // Hypergraph<T, W>: HypergraphAccessor<4, T, W>,
// {
//     fn from(rep: Hypergraph<T, W>) -> Self {
//         let mut degree = [0u32; 4];
//
//         seq!(M in 2..5 {
//                 rep.edges::<M>().into_iter().for_each(|edge| {
//                     for n in edge{
//                         // let u = edge.nodes[i].as_() as usize;
//                         degree[n.as_() as usize] += 1;
//                     }
//                 });
//         });
//
//         degree.sort_unstable();
//
//         Self {
//             degree,
//             order_map: None,
//             rep: rep.into_unweighted(),
//         }
//     }
// }

impl<T, W> From<Hypergraph<T, W>> for CanonicalRep<5, T>
where
    T: Shl + AsPrimitive<u8> + PrimInt + Hash,
    // Hypergraph<T, W>: HypergraphAccessor<4, T, W>,
{
    fn from(rep: Hypergraph<T, W>) -> Self {
        let mut degree = [0u32; 5];

        seq!(M in 2..5 {
                rep.edges::<M>().into_iter().for_each(|edge| {
                    for n in edge{
                        // let u = edge.nodes[i].as_() as usize;
                        degree[n.as_() as usize] += 1;
                    }
                });
        });

        degree.sort_unstable();

        Self {
            degree,
            order_map: None,
            rep: rep.into_unweighted(),
        }
    }
}

// pub fn from1(rep: Hypergraph<T, ()>) -> Self {
//     // 1. Collect all local hyperedges as compact 5-bit masks on the stack.
//     // In an order-5 subhypergraph, max possible edges with size >= 2 is 26.
//     let mut local_edges = [0u8; 26];
//     let mut edge_idx = 0;
//
//     seq!(N in 2..6 {
//         rep.edges::<N>().iter().for_each(|edge| {
//             let mut compact_edge: u8 = 0;
//             for n in edge {
//                 compact_edge |= 1 << n.as_();
//             }
//
//             // Protect against stack overflow, though mathematically capped at 26
//             if edge_idx < local_edges.len() {
//                 local_edges[edge_idx] = compact_edge;
//                 edge_idx += 1;
//             }
//         });
//     });
//
//     // We only care about the active slice of edges found
//     let active_edges = &local_edges[0..edge_idx];
//
//     // 2. Compute the exact multi-set intersection profiles for each vertex
//     let mut vertex_scores = [0u32; N];
//
//     for u in 0..N {
//         let mut intersections = [0u8; 4]; // Distances to the other 4 nodes
//         let mut idx = 0;
//
//         for v in 0..N {
//             if u == v {
//                 continue;
//             }
//
//             // This mask guarantees both node 'u' and node 'v' are present
//             let pair_mask = (1 << u) | (1 << v);
//             let mut shared_hyperedges_count = 0u8;
//
//             for &edge in active_edges {
//                 if (edge & pair_mask) == pair_mask {
//                     shared_hyperedges_count += 1;
//                 }
//             }
//
//             if idx < 4 {
//                 intersections[idx] = shared_hyperedges_count;
//                 idx += 1;
//             }
//         }
//
//         intersections.sort_unstable();
//
//         // Compute vertex degree within this local subhypergraph
//         let mut degree = 0u8;
//         let u_mask = 1 << u;
//         for &edge in active_edges {
//             if (edge & u_mask) != 0 {
//                 degree += 1;
//             }
//         }
//
//         // Pack the degree and sorted intersections into a clean u32 role score
//         vertex_scores[u] = ((degree as u32) << 24)
//             | ((intersections[0] as u32) << 16)
//             | ((intersections[1] as u32) << 8)
//             | (intersections[2] as u32);
//     }
//
//     Self {
//         scores: vertex_scores,
//         rep,
//     }
//     // From here, proceed with your permutation mapping using vertex_scores...
// }

// #[pyclass]
// pub struct MotifStat {
//     pub canonical_rep: CanonicalRep<NodeId>,
//     pub count: usize,
//     pub intensity: f32,
//     pub coherence: Option<f32>,
//     pub actual_coherence: Option<f32>,
// }
