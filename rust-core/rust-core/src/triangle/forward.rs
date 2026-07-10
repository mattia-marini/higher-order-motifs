use std::ops::Deref;

use crate::triangle::cbs::hcbs::HCBSGraph;

use crate::misc::{count_common_neighbors_sorted_list, degeneracy_ordering, degree_ordering};
use crate::types::NodeId;
use crate::types::adj_list::AdjList;
use crate::types::adj_list::common::Undirected;
use crate::types::adj_list::traits::Incidence;

/// Computes the degree ordering: (order, position)

#[cfg(feature = "bindings")]
#[pyo3::pymodule(submodule)]
pub mod forward {
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;
    use pyo3_stub_gen::reexport_module_members;

    use crate::types::adj_list::{PyAdjList, PyUndirectedAdjList};

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.triangle.forward")]
    pub fn forward(adj: PyUndirectedAdjList, sort_degrees: bool) -> usize {
        match adj {
            PyUndirectedAdjList::Weighted(g) => super::forward(&g, sort_degrees),
            PyUndirectedAdjList::Unweighted(g) => super::forward(&g, sort_degrees),
        }
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.triangle.forward")]
    pub fn forward_hashed(adj: PyUndirectedAdjList, sort_degrees: bool) -> usize {
        match adj {
            PyUndirectedAdjList::Weighted(g) => super::forward(&g, sort_degrees),
            PyUndirectedAdjList::Unweighted(g) => super::forward(&g, sort_degrees),
        }
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.triangle.forward")]
    pub fn forward_hbs(adj: PyUndirectedAdjList, sort_degrees: bool) -> usize {
        match adj {
            PyUndirectedAdjList::Weighted(g) => super::forward(&g, sort_degrees),
            PyUndirectedAdjList::Unweighted(g) => super::forward(&g, sort_degrees),
        }
    }

    reexport_module_members!("rust_core.triangle.forward" from "rust_core._core.triangle.forward");
}

/// Forward algorithm for triangle counting. If sort_degrees is true, a degree ordering is computed, otherwise edges are processed in
/// the natural order (u < v). Common neighbors are counted with the sorted list strategy
pub fn forward<W, I: Incidence>(adj: &AdjList<W, Undirected, I>, sort_degrees: bool) -> usize {
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
            for neighbor in &adj[u] {
                // Using Index trait
                let v = neighbor.node as usize;
                // let w = neighbor.weight.clone();
                if i < pos[v] as usize {
                    // a[u] works if u is usize. If u is NodeId, use u as usize
                    count += count_common_neighbors_sorted_list(&a[u as usize], &a[v]);
                    a[v].push((pos[u as usize] as NodeId, ())); // Cast back to NodeId for storage
                }
            }
        }
    } else {
        for u in 0..n {
            for neighbor in &adj[u] {
                // &(v_node, ref w)
                let v = neighbor.node as usize;
                // let w = neighbor.weight.clone();
                if u < v {
                    count += count_common_neighbors_sorted_list(&a[u], &a[v]);
                    a[v].push((u as NodeId, ()));
                }
            }
        }
    }
    count
}

/// Compact forward/forward hashed algorithm for triangle counting. If sort_degrees is true, a degree ordering is computed, otherwise edges are processed in
/// the natural order (u < v). Common neighbors are counted with the hash map strategy
pub fn forward_hashed<W, I: Incidence>(
    adj: &AdjList<W, Undirected, I>,
    order: Option<(&[NodeId], &[usize])>,
) -> usize {
    let n = adj.n();
    let mut a = vec![Vec::new(); n];
    let mut mark = vec![0usize; n];
    let mut current = 1;
    let mut count = 0;

    let natural_order = ((0 as NodeId)..(n as NodeId)).collect::<Vec<_>>();
    let natural_pos = (0..n).collect::<Vec<_>>();
    let (order, pos) = match order {
        Some((o, p)) => (o, p),
        None => (natural_order.deref(), natural_pos.deref()),
    };

    for i in 0..n {
        let u = order[i] as usize; // Cast once per outer loop

        for neighbor in &adj[u] {
            // &(v_node, ref _w)
            let v = neighbor.node as usize;
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

pub fn forward_hbs<W, I: Incidence>(adj: &AdjList<W, Undirected, I>, sort_degrees: bool) -> usize {
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
            for neighbor in &adj[u_idx] {
                let v = neighbor.node as usize;
                if i < pos[v] as usize {
                    count += a.count_common_neighbors(u_idx, v);
                    a.append_neighbor(v as NodeId, pos[u_idx] as NodeId);
                }
            }
        }
    } else {
        for u in 0..n {
            for neighbor in &adj[u] {
                let v = neighbor.node as usize;
                if u < v {
                    count += a.count_common_neighbors(u, v);
                    a.append_neighbor(v as NodeId, u as NodeId);
                }
            }
        }
    }
    count
}

/// the order parameter specifies order and position array for the nodes. A vertex degree order or
/// degeneracy order can be used. If None, the natural order is used
pub fn forward_hashed_cloj<W, I, F>(
    adj: &AdjList<W, Undirected, I>,
    order: Option<(&[NodeId], &[usize])>,
    mut cloj: F,
) where
    F: FnMut(NodeId, NodeId, NodeId),
    I: Incidence,
{
    let n = adj.n();
    let mut a = vec![Vec::new(); n];
    let mut mark = vec![0usize; n];
    let mut current = 1;

    let natural_order = ((0 as NodeId)..(n as NodeId)).collect::<Vec<_>>();
    let natural_pos = (0..n).collect::<Vec<_>>();
    let (order, pos) = match order {
        Some((o, p)) => (o, p),
        None => (natural_order.deref(), natural_pos.deref()),
    };

    for i in 0..n {
        let u = order[i] as usize; // Cast once per outer loop

        for neighbor in &adj[u] {
            let v = neighbor.node as usize;
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
