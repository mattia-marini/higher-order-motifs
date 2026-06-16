use crate::triangle::cbs::hcbs::HCBSGraph;

use crate::graph::{AdjList, types::NodeId};
use crate::misc::{count_neighbors_sorted_list, degree_ordering};
use pyo3::prelude::*;
use pyo3_stub_gen::reexport_module_members;

/// Computes the degree ordering: (order, position)

#[pymodule(submodule)]
pub mod forward {
    use crate::graph::{AdjList, PyAdjList};
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.triangle.forward")]
    pub fn forward(adj: PyAdjList, sort_degrees: bool) -> usize {
        match adj {
            PyAdjList::Unweighted(g) => super::forward(&g, sort_degrees),
            PyAdjList::Weighted(g) => super::forward(&g, sort_degrees),
        }
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.triangle.forward")]
    pub fn forward_hashed(adj: PyAdjList, sort_degrees: bool) -> usize {
        match adj {
            PyAdjList::Unweighted(g) => super::forward_hashed(&g, sort_degrees),
            PyAdjList::Weighted(g) => super::forward_hashed(&g, sort_degrees),
        }
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.triangle.forward")]
    pub fn forward_hbs(adj: PyAdjList, sort_degrees: bool) -> usize {
        match adj {
            PyAdjList::Unweighted(g) => super::forward_hbs(&g, sort_degrees),
            PyAdjList::Weighted(g) => super::forward_hbs(&g, sort_degrees),
        }
    }
}

/// Forward algorithm for triangle counting. If sort_degrees is true, a degree ordering is computed, otherwise edges are processed in
/// the natural order (u < v). Common neighbors are counted with the sorted list strategy
pub fn forward<W>(adj: &AdjList<W>, sort_degrees: bool) -> usize {
    let n = adj.n();
    let mut a = vec![Vec::new(); n];
    for i in 0..n {
        a[i].reserve(adj[i].len());
    }

    let mut count = 0;

    if sort_degrees {
        let (order, pos, _) = degree_ordering(adj, true);

        for i in 0..n {
            let u = order[i]; // order usually contains NodeId
            for &(v_node, ref w) in &adj[u] {
                // Using Index trait
                let v = v_node as usize;
                if i < pos[v] as usize {
                    // a[u] works if u is usize. If u is NodeId, use u as usize
                    count += count_neighbors_sorted_list(&a[u as usize], &a[v]);
                    a[v].push((pos[u as usize] as NodeId, w)); // Cast back to NodeId for storage
                }
            }
        }
    } else {
        for u in 0..n {
            for &(v_node, ref w) in &adj[u] {
                let v = v_node as usize;
                if u < v {
                    count += count_neighbors_sorted_list(&a[u], &a[v]);
                    a[v].push((u as NodeId, w));
                }
            }
        }
    }
    count
}

/// Compact forward/forward hashed algorithm for triangle counting. If sort_degrees is true, a degree ordering is computed, otherwise edges are processed in
/// the natural order (u < v). Common neighbors are counted with the hash map strategy
pub fn forward_hashed<W>(adj: &AdjList<W>, sort_degrees: bool) -> usize {
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

        for &(v_node, ref _w) in &adj[u] {
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

pub fn forward_hbs<W>(adj: &AdjList<W>, sort_degrees: bool) -> usize {
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
            for &(v_node, ref _w) in &adj[u_idx] {
                let v = v_node as usize;
                if i < pos[v] as usize {
                    count += a.count_common_neighbors(u_idx, v);
                    a.append_neighbor(v as NodeId, pos[u_idx] as NodeId);
                }
            }
        }
    } else {
        for u in 0..n {
            for &(v_node, ref _w) in &adj[u] {
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

pub fn forward_hashed_cloj<W, F>(adj: &AdjList<W>, sort_degrees: bool, mut cloj: F)
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

        for &(v_node, ref _w) in &adj[u] {
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

reexport_module_members!("rust_core.triangle.forward" from "rust_core._core.triangle.forward");
