use crate::triangle::cbs::hcbs::HCBSGraph;

use super::common::{count_neighbors_sorted_list, degree_ordering};
use bit_set::BitSet;
use pyo3::prelude::*;
use pyo3_stub_gen::reexport_module_members;

/// Computes the degree ordering: (order, position)

#[pymodule(submodule)]
pub mod forward {
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.forward")]
    pub fn forward(adj: Vec<Vec<usize>>, sort_degrees: bool) -> usize {
        super::forward(&adj, sort_degrees)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.forward")]
    pub fn forward_hashed(adj: Vec<Vec<usize>>, sort_degrees: bool) -> usize {
        super::forward_hashed(&adj, sort_degrees)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.forward")]
    pub fn forward_hbs(adj: Vec<Vec<usize>>, sort_degrees: bool) -> usize {
        super::forward_hbs(&adj, sort_degrees)
    }
}

/// Forward algorithm for triangle counting. If sort_degrees is true, a degree ordering is computed, otherwise edges are processed in
/// the natural order (u < v). Common neighbors are counted with the sorted list strategy
pub fn forward(adj: &Vec<Vec<usize>>, sort_degrees: bool) -> usize {
    let n = adj.len();
    let mut a = vec![Vec::new(); n];
    for i in 0..n {
        a[i].reserve(adj[i].len());
    }

    let mut count = 0;

    if sort_degrees {
        let (order, pos, _) = degree_ordering(adj);

        for i in 0..n {
            let u = order[i];
            for &v in &adj[u] {
                // Only process directed edge u -> v in the DAG
                if i < pos[v] {
                    // Count common neighbors in the 'a' sets
                    count += count_neighbors_sorted_list(&a[u], &a[v]);
                    a[v].push(pos[u]);
                }
            }
        }
    } else {
        for u in 0..n {
            for &v in &adj[u] {
                if u < v {
                    count += count_neighbors_sorted_list(&a[u], &a[v]);
                    a[v].push(u);
                }
            }
        }
    }
    count
}

/// Compact forward/forward hashed algorithm for triangle counting. If sort_degrees is true, a degree ordering is computed, otherwise edges are processed in
/// the natural order (u < v). Common neighbors are counted with the hash map strategy
pub fn forward_hashed(adj: &Vec<Vec<usize>>, sort_degrees: bool) -> usize {
    let n = adj.len();
    let mut a = vec![Vec::new(); n];
    for i in 0..n {
        a[i].reserve(adj[i].len());
    }

    let mut bitset = BitSet::with_capacity(n);
    let mut count = 0;

    let (order, pos, _) = if sort_degrees {
        degree_ordering(adj)
    } else {
        (vec![], vec![], 0)
    };

    for i in 0..n {
        let u = if sort_degrees { order[i] } else { i };
        for &v in &adj[u] {
            let is_forward = if sort_degrees { i < pos[v] } else { u < v };

            if is_forward {
                // Mark elements in a[u]
                for &w in &a[u] {
                    bitset.insert(w);
                }
                // Check elements in a[v] against the mark
                for &w in &a[v] {
                    if bitset.contains(w) {
                        count += 1;
                    }
                }
                // Reset marks for next iteration
                for &w in &a[u] {
                    bitset.remove(w);
                }
                a[v].push(u);
            }
        }
    }
    count
}

/// Forward algorithm for triangle counting. If sort_degrees is true, a degree ordering is computed, otherwise edges are processed in
/// the natural order (u < v). Common neighbors are counted with an optimized hierarchical bitset
pub fn forward_hbs(adj: &Vec<Vec<usize>>, sort_degrees: bool) -> usize {
    let n = adj.len();
    let mut a = HCBSGraph::<u128>::with_nodes(n);
    for i in 0..n {
        a.nodes[i].bits.reserve(adj[i].len());
        a.nodes[i].offsets.reserve(adj[i].len());
    }

    let mut count = 0;

    if sort_degrees {
        let (order, pos, _) = degree_ordering(adj);

        for i in 0..n {
            let u = order[i];
            for &v in &adj[u] {
                // Only process directed edge u -> v in the DAG
                if i < pos[v] {
                    // Count common neighbors in the 'a' sets
                    count += a.count_common_neighbors(u, v);
                    a.append_neighbor(v, pos[u]);
                }
            }
        }
    } else {
        for u in 0..n {
            for &v in &adj[u] {
                if u < v {
                    count += a.count_common_neighbors(u, v);
                    a.append_neighbor(v, u);
                }
            }
        }
    }
    count
}

reexport_module_members!("rust_core.triangle.forward" from "rust_core.core.triangle.forward");
