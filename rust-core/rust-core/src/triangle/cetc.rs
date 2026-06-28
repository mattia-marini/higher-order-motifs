use crate::graph::AdjList;
use crate::graph::hyperedge::NodeId;
use crate::misc::{bfs, common_neighbors_sorted_list};

#[cfg(feature = "bindings")]
#[pyo3::pymodule(submodule)]
pub mod cetc {
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;
    use pyo3_stub_gen::reexport_module_members;

    use crate::graph::{AdjList, PyAdjList};

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.triangle.cetc")]
    pub fn cetc(adj: PyAdjList) -> usize {
        match adj {
            PyAdjList::Weighted(g) => super::cetc(&mut g.clone()),
            PyAdjList::Unweighted(g) => super::cetc(&mut g.clone()),
        }
    }

    #[gen_stub_pyfunction(module = "rust_core._core.triangle.cetc")]
    #[pyfunction]
    pub fn cetc_s(adj: PyAdjList) -> usize {
        match adj {
            PyAdjList::Weighted(g) => super::cetc_s(&g),
            PyAdjList::Unweighted(g) => super::cetc_s(&g),
        }
    }

    reexport_module_members!("rust_core.triangle.cetc" from "rust_core._core.triangle.cetc");
}

/// Computes intersection of two sorted vectors and returns the common elements.
pub fn cetc<W>(adj: &mut AdjList<W>) -> usize
where
    W: Clone,
{
    let n = adj.n();
    let mut count = 0;

    adj.sort_neighbors();
    let levels = bfs(adj);

    for u in 0..n {
        for &(v_node, ref _w) in &adj.adj[u] {
            let v = v_node as usize;
            // Check levels and use u < v to avoid double counting
            if levels[v] == levels[u] && u < v {
                let common = common_neighbors_sorted_list(&adj.adj[u], &adj.adj[v]);
                for w in common {
                    // Triangle (u, v, w) logic
                    if levels[w.0 as usize] != levels[u] || v < w.0 as usize {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

pub fn cetc_s<W>(adj: &AdjList<W>) -> usize
where
    W: Clone,
{
    let n = adj.n();
    let mut adj0 = vec![vec![]; n];
    let mut adj1 = vec![vec![]; n];
    let mut hash = vec![false; n];
    let mut count = 0;

    let levels = bfs(adj);

    // Partition edges based on levels
    for u in 0..n {
        for v in adj.adj[u].clone() {
            if levels[u] == levels[v.0 as usize] {
                adj0[u].push(v);
            } else {
                adj1[u].push(v);
            }
        }
    }

    // Reuse the compact_forward from previous implementation
    // Build an AdjList from adj0 to call forward_hashed
    let mut al0 = AdjList::with_nodes(n);
    for u in 0..n {
        for v in adj0[u].clone() {
            al0.adj[u].push(v);
        }
    }
    count += super::forward::forward_hashed(&al0, false);

    for u in 0..n {
        if adj1[u].is_empty() {
            continue;
        }

        // Standard hash-based intersection logic
        for &(v, ref _w) in &adj1[u] {
            hash[v as usize] = true;
        }

        for &(v, ref _w) in &adj0[u] {
            let v = v as usize;
            if u < v {
                for &(w, ref _w) in &adj1[v] {
                    let w = w as usize;
                    if hash[w] {
                        count += 1;
                    }
                }
            }
        }

        // Clean up hash for the next iteration
        for &(v, ref _w) in &adj1[u] {
            hash[v as usize] = false;
        }
    }

    count
}
