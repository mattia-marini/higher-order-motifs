use num_traits::{AsPrimitive, PrimInt};

use crate::types::hyperadj_list::HyperAdjList;
use crate::types::{Hx, Hypergraph, NodeId};
use std::hash::Hash;

use crate::types::adj_list::AdjList;
use crate::types::adj_list::traits::{Direction, Incidence};
use crate::types::adj_list::{
    adj_list::AdjListBase,
    traits::{AdjConfig, NeighborContainer},
};

/// Returns a degree ordering of the vertices, the position of each vertex in that ordering, and
/// the maximum degree of the graph.
/// Time Complexity: O(n)
pub fn degree_ordering<C: AdjConfig>(
    adj: &AdjListBase<C>,
    decreasing: bool,
) -> (Vec<NodeId>, Vec<usize>, usize) {
    let n = adj.n();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0);
    }

    let deg: Vec<usize> = adj
        .iter_neighbors()
        .map(|neighbors| neighbors.len())
        .collect();
    let max_deg = *deg.iter().max().unwrap_or(&0);

    // Count how many vertices have each degree
    let mut bin_count = vec![0; max_deg + 1];
    for &d in &deg {
        bin_count[d] += 1;
    }

    // Compute starting index for each degree bin
    let mut start_pos = 0;
    let mut bin_starts = vec![0; max_deg + 1];
    for d in 0..=max_deg {
        bin_starts[d] = start_pos;
        start_pos += bin_count[d];
    }

    // Fill order and pos
    let mut order = vec![0; n];
    let mut pos = vec![0; n];

    if decreasing {
        for v in (0..n).rev() {
            let d = deg[v];
            pos[v] = bin_starts[d];
            order[bin_starts[d]] = v as NodeId;
            bin_starts[d] += 1;
        }
    } else {
        for v in 0..n {
            let d = deg[v];
            pos[v] = bin_starts[d];
            order[bin_starts[d]] = v as NodeId;
            bin_starts[d] += 1;
        }
    }

    (order, pos, max_deg)
}

/// Sorts the neighbors of each vertex in the adjacency list by the following conditions:
/// u ≺ v if deg(u) < deg(v); if deg(u) = deg(v) the tie breaker is arbitrary
///
/// Time Complexity: O(e log d), where e is the number of edges and d is the maximum degree.
pub fn sort_by_degree<W, D: Direction, I: Incidence>(
    adj: &mut AdjList<W, D, I>,
    descreasing: bool,
) -> (Vec<NodeId>, Vec<usize>, usize) {
    let (order, rank, max_deg) = degree_ordering(adj, false);

    for v in 0..adj.n() {
        adj[v].sort_by_key(|neighbor| rank[neighbor.node as usize]);
    }

    // adj.adj.enumerate().sort_by_key(|(i, v)| rank[*i]);
    (order, rank, max_deg)
}

/// Returns a degeneracy ordering of the graph, the position of each vertex,
/// and the degeneracy (k) of the graph.
/// Complexity: O(n + m)
pub fn degeneracy_ordering<C: AdjConfig>(adj: &AdjListBase<C>) -> (Vec<NodeId>, Vec<usize>, usize) {
    let n = adj.n();
    if n == 0 {
        return (vec![], vec![], 0);
    }

    // 1. Calculate degrees and find max degree
    let mut deg: Vec<usize> = adj
        .iter_neighbors()
        .map(|neighbors| neighbors.len())
        .collect();
    let max_deg = *deg.iter().max().unwrap_or(&0);

    // 2. Create bins to count how many nodes have each degree
    let mut bin_count = vec![0; max_deg + 1];
    for &d in &deg {
        bin_count[d] += 1;
    }

    // 3. Find starting index for each degree bucket
    let mut bin_starts = vec![0; max_deg + 1];
    let mut start_pos = 0;
    for d in 0..=max_deg {
        bin_starts[d] = start_pos;
        start_pos += bin_count[d];
    }

    // 4. Initial placement of nodes into 'order' and 'pos'
    let mut temp_starts = bin_starts.clone();
    let mut order = vec![0; n];
    let mut pos = vec![0; n];
    for v in 0..n {
        pos[v] = temp_starts[deg[v]];
        order[pos[v]] = v as NodeId;
        temp_starts[deg[v]] += 1;
    }

    // 5. Main loop: remove node of minimum degree
    let mut k = 0;
    macro_rules! decrease_node {
        ($node:expr) => {{
            unsafe {
                let n = $node;
                let u_deg = *deg.get_unchecked(n);
                let u_pos = *pos.get_unchecked(n);

                let first_node_pos = *bin_starts.get_unchecked(u_deg);
                let first_node = *order.get_unchecked(first_node_pos);

                if n as NodeId != first_node {
                    // pos.swap(n, first_node);
                    let tmp = *pos.get_unchecked(first_node as usize);
                    *pos.get_unchecked_mut(first_node as usize) = *pos.get_unchecked(n);
                    *pos.get_unchecked_mut(n) = tmp;

                    // order.swap(u_pos, first_node_pos);
                    let tmp = *order.get_unchecked(first_node_pos);
                    *order.get_unchecked_mut(first_node_pos) = *order.get_unchecked(u_pos);
                    *order.get_unchecked_mut(u_pos) = tmp;
                }

                *bin_starts.get_unchecked_mut(u_deg) += 1;
                *deg.get_unchecked_mut(n) -= 1;
            }
        }};
    }

    for i in 0..n {
        let v = order[i] as usize;
        k = std::cmp::max(k, deg[v]);

        for neighbor in adj[v].iter_neighbors() {
            let u = *neighbor.node as usize;
            if pos[u] > i {
                decrease_node!(u);
                decrease_node!(v);
            }
        }
    }

    (order, pos, k)
}

/// Returns a degeneracy ordering of the hypergraph, the position of each vertex,
/// and the degeneracy (k) of the hypergraph.
/// Complexity: O(n + m)
pub fn hyper_degeneracy_ordering<W>(adj: &HyperAdjList<W>) -> (Vec<usize>, Vec<usize>, usize) {
    let n = adj.n();
    if n == 0 {
        return (vec![], vec![], 0);
    }

    // 1. Calculate degrees and find max degree
    let mut deg = adj
        .adj
        .iter()
        .map(|neighbors| neighbors.len())
        .collect::<Vec<_>>();
    let max_deg = *deg.iter().max().unwrap_or(&0);

    // 2. Create bins to count how many nodes have each degree
    let mut bin_count = vec![0; max_deg + 1];
    for &d in &deg {
        bin_count[d] += 1;
    }

    // 3. Find starting index for each degree bucket
    let mut bin_starts = vec![0; max_deg + 1];
    let mut start_pos = 0;
    for d in 0..=max_deg {
        bin_starts[d] = start_pos;
        start_pos += bin_count[d];
    }

    // 4. Initial placement of nodes into 'order' and 'pos'
    let mut temp_starts = bin_starts.clone();
    let mut order = vec![0; n];
    let mut pos = vec![0; n];
    for v in 0..n {
        pos[v] = temp_starts[deg[v]];
        order[pos[v]] = v;
        temp_starts[deg[v]] += 1;
    }

    let mut peeled = vec![false; adj.m()];
    // 5. Main loop: remove node of minimum degree
    let mut k = 0;
    for i in 0..n {
        let v = order[i];
        k = std::cmp::max(k, deg[v]);

        for (edge_id, edge) in adj.iter_incident_edges(v as NodeId) {
            if peeled[edge_id as usize] {
                continue;
            }
            peeled[edge_id as usize] = true;

            for &n in edge.nodes {
                // if n == v as NodeId {
                //     continue;
                // }
                let u = n as usize;
                let u_deg = deg[u];
                let u_pos = pos[u];

                // The first node in u's degree bucket
                let first_node_pos = bin_starts[u_deg];
                let first_node = order[first_node_pos];

                // Swap u with the first node in its bucket
                if u != first_node {
                    pos.swap(u, first_node);
                    order.swap(u_pos, first_node_pos);
                }

                // Move the bucket boundary forward and decrease degree
                bin_starts[u_deg] += 1;
                deg[u] -= 1;
            }
        }
    }

    (order, pos, k)
}

// A version of degeneracy_ordering that accepts Python objects.
// It maps Python objects to internal indices to perform the O(n + m) sort.
// #[cfg(feature = "bindings")]
// pub fn degeneracy_ordering_py<W>(
//     adj: &AdjList<W>,
// ) -> pyo3::prelude::PyResult<(Vec<usize>, Vec<usize>, usize)> {
//     let n = adj.n();
//     if n == 0 {
//         return Ok((vec![], vec![], 0));
//     }
//
//     let mut deg: Vec<usize> = adj.adj.iter().map(|neighbors| neighbors.len()).collect();
//     let max_deg = *deg.iter().max().unwrap_or(&0);
//
//     let mut bin_count = vec![0; max_deg + 1];
//     for &d in &deg {
//         bin_count[d] += 1;
//     }
//
//     let mut bin_starts = vec![0; max_deg + 1];
//     let mut start_pos = 0;
//     for d in 0..=max_deg {
//         bin_starts[d] = start_pos;
//         start_pos += bin_count[d];
//     }
//
//     let mut temp_starts = bin_starts.clone();
//     let mut order_idx = vec![0; n];
//     let mut pos = vec![0; n];
//     for v in 0..n {
//         pos[v] = temp_starts[deg[v]];
//         order_idx[pos[v]] = v;
//         temp_starts[deg[v]] += 1;
//     }
//
//     let mut k = 0;
//     for i in 0..n {
//         let v = order_idx[i];
//         k = std::cmp::max(k, deg[v]);
//         for &(u_node, ref _w) in &adj.adj[v] {
//             let u = u_node as usize;
//             if pos[u] > i {
//                 let u_deg = deg[u];
//                 let u_pos = pos[u];
//                 let first_node_pos = bin_starts[u_deg];
//                 let first_node = order_idx[first_node_pos];
//
//                 if u != first_node {
//                     pos.swap(u, first_node);
//                     order_idx.swap(u_pos, first_node_pos);
//                 }
//
//                 bin_starts[u_deg] += 1;
//                 deg[u] -= 1;
//             }
//         }
//     }
//
//     Ok((order_idx, pos, k))
// }
