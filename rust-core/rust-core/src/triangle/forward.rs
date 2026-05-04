use crate::triangle::cbs::hcbs::HCBSGraph;

use crate::graph::{AdjList, types::NodeId};
use crate::misc::{count_neighbors_sorted_list, degree_ordering};
use pyo3::prelude::*;
use pyo3_stub_gen::reexport_module_members;

/// Computes the degree ordering: (order, position)

#[pymodule(submodule)]
pub mod forward {
    use crate::graph::AdjList;
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.forward")]
    pub fn forward(adj: &AdjList, sort_degrees: bool) -> usize {
        super::forward(adj, sort_degrees)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.forward")]
    pub fn forward_hashed(adj: &AdjList, sort_degrees: bool) -> usize {
        super::forward_hashed(adj, sort_degrees)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.forward")]
    pub fn forward_hbs(adj: &AdjList, sort_degrees: bool) -> usize {
        super::forward_hbs(adj, sort_degrees)
    }
}

// pub fn _forward_hashed(adj: &AdjList, sort_degrees: bool) -> usize {
//     let n = adj.n();
//     let mut a = vec![Vec::new(); n];
//     for i in 0..n {
//         a[i].reserve(adj.adj[i].len());
//     }
//
//     let mut bitset = BitSet::with_capacity(n);
//     let mut count = 0;
//
//     let (order, pos, _) = if sort_degrees {
//         degree_ordering(adj, true)
//     } else {
//         (vec![], vec![], 0)
//     };
//
//     for i in 0..n {
//         let u = if sort_degrees { order[i] } else { i };
//         for &v_node in &adj.adj[u] {
//             let v = v_node as usize;
//             let is_forward = if sort_degrees { i < pos[v] } else { u < v };
//
//             if is_forward {
//                 // Mark elements in a[u]
//                 for &w in &a[u] {
//                     bitset.insert(w);
//                 }
//                 // Check elements in a[v] against the mark
//                 for &w in &a[v] {
//                     if bitset.contains(w) {
//                         count += 1;
//                     }
//                 }
//                 // Reset marks for next iteration
//                 for &w in &a[u] {
//                     bitset.remove(w);
//                 }
//                 a[v].push(u);
//             }
//         }
//     }
//     count
// }

/// Forward algorithm for triangle counting. If sort_degrees is true, a degree ordering is computed, otherwise edges are processed in
/// the natural order (u < v). Common neighbors are counted with the sorted list strategy
pub fn forward(adj: &AdjList, sort_degrees: bool) -> usize {
    let n = adj.n();
    let mut a = vec![Vec::new(); n];
    for i in 0..n {
        a[i].reserve(adj[i].len()); // Using Index trait
    }

    let mut count = 0;

    if sort_degrees {
        let (order, pos, _) = degree_ordering(adj, true);

        for i in 0..n {
            let u = order[i]; // order usually contains NodeId
            for &v_node in &adj[u] {
                // Using Index trait
                let v = v_node as usize;
                if i < pos[v] as usize {
                    // a[u] works if u is usize. If u is NodeId, use u as usize
                    count += count_neighbors_sorted_list(&a[u as usize], &a[v]);
                    a[v].push(pos[u as usize] as NodeId); // Cast back to NodeId for storage
                }
            }
        }
    } else {
        for u in 0..n {
            for &v_node in &adj[u] {
                let v = v_node as usize;
                if u < v {
                    count += count_neighbors_sorted_list(&a[u], &a[v]);
                    a[v].push(u as NodeId);
                }
            }
        }
    }
    count
}

/// Compact forward/forward hashed algorithm for triangle counting. If sort_degrees is true, a degree ordering is computed, otherwise edges are processed in
/// the natural order (u < v). Common neighbors are counted with the hash map strategy
pub fn forward_hashed(adj: &AdjList, sort_degrees: bool) -> usize {
    let n = adj.n();
    let mut a = vec![Vec::new(); n];
    let mut mark = vec![0usize; n];
    let mut current = 1;
    let mut count = 0;

    let (order, pos) = if sort_degrees {
        let (o, p, _) = degree_ordering(adj, true);
        (o, p)
    } else {
        ((0..n as NodeId).collect(), (0..n as NodeId).collect())
    };

    for i in 0..n {
        let u = order[i] as usize; // Cast once per outer loop

        for &v_node in &adj[u] {
            let v = v_node as usize;
            let is_forward = i < pos[v] as usize;

            if is_forward {
                for &w in &a[u] {
                    mark[w as usize] = current;
                }

                for &w in &a[v] {
                    if mark[w as usize] == current {
                        count += 1;
                    }
                }

                current += 1;
                a[v].push(u as NodeId);
            }
        }
    }
    count
}

pub fn forward_hbs(adj: &AdjList, sort_degrees: bool) -> usize {
    let n = adj.n();
    let mut a = HCBSGraph::<u128>::with_nodes(n);

    // Optimization: Pre-reserve
    for i in 0..n {
        a.nodes[i].bits.reserve(adj[i].len());
        a.nodes[i].offsets.reserve(adj[i].len());
    }

    let mut count = 0;

    if sort_degrees {
        let (order, pos, _) = degree_ordering(adj, true);

        for i in 0..n {
            let u = order[i];
            let u_idx = u as usize;
            for &v_node in &adj[u_idx] {
                let v = v_node as usize;
                if i < pos[v] as usize {
                    count += a.count_common_neighbors(u_idx, v);
                    a.append_neighbor(v as NodeId, pos[u_idx] as NodeId);
                }
            }
        }
    } else {
        for u in 0..n {
            for &v_node in &adj[u] {
                let v = v_node as usize;
                if u < v {
                    count += a.count_common_neighbors(u, v);
                    a.append_neighbor(v as NodeId, u as NodeId);
                }
            }
        }
    }
    count
}

pub fn forward_hashed_cloj<F>(adj: &AdjList, sort_degrees: bool, mut cloj: F)
where
    F: FnMut(NodeId, NodeId, NodeId),
{
    let n = adj.n();
    let mut a = vec![Vec::new(); n];
    let mut mark = vec![0usize; n];
    let mut current = 1;

    let (order, pos) = if sort_degrees {
        let (o, p, _) = degree_ordering(adj, true);
        (o, p)
    } else {
        ((0..n as NodeId).collect(), (0..n as NodeId).collect())
    };

    for i in 0..n {
        let u = order[i] as usize; // Cast once per outer loop

        for &v_node in &adj[u] {
            let v = v_node as usize;
            let is_forward = i < pos[v] as usize;

            if is_forward {
                for &w in &a[u] {
                    mark[w as usize] = current;
                }

                for &w in &a[v] {
                    if mark[w as usize] == current {
                        cloj(u as NodeId, v as NodeId, w);
                    }
                }

                current += 1;
                a[v].push(u as NodeId);
            }
        }
    }
}

reexport_module_members!("rust_core.triangle.forward" from "rust_core.core.triangle.forward");
