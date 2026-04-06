use super::common::bfs;
use pyo3::prelude::*;

#[pymodule(submodule)]
pub mod cetc {
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    #[gen_stub_pyfunction(module = "rust_core._core.triangle.cetc")]
    #[pyfunction]
    pub fn cetc(mut adj: Vec<Vec<usize>>) -> usize {
        super::cetc(&mut adj)
    }

    #[gen_stub_pyfunction(module = "rust_core._core.triangle.cetc")]
    #[pyfunction]
    pub fn cetc_s(adj: Vec<Vec<usize>>) -> usize {
        super::cetc_s(&adj)
    }
}


/// Computes intersection of two sorted vectors and returns the common elements.

pub fn cetc(adj: &mut Vec<Vec<usize>>) -> usize {
    let n = adj.len();
    let mut count = 0;

    super::common::sort_adj_list(adj);
    let levels = bfs(&adj);

    for u in 0..n {
        for &v in &adj[u] {
            // Check levels and use u < v to avoid double counting
            if levels[v] == levels[u] && u < v {
                let common = super::common::common_neighbors_sorted_list(&adj[u], &adj[v]);
                for w in common {
                    // Triangle (u, v, w) logic
                    if levels[w] != levels[u] || v < w {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

pub fn cetc_s(adj: &Vec<Vec<usize>>) -> usize {
    let n = adj.len();
    let mut adj0 = vec![vec![]; n];
    let mut adj1 = vec![vec![]; n];
    let mut hash = vec![false; n];
    let mut count = 0;

    let levels = bfs(adj);

    // Partition edges based on levels
    for u in 0..n {
        for &v in &adj[u] {
            if levels[u] == levels[v] {
                adj0[u].push(v);
            } else {
                adj1[u].push(v);
            }
        }
    }

    // Reuse the compact_forward from previous implementation
    count += super::forward::compact_forward(&adj0, false);

    for u in 0..n {
        if adj1[u].is_empty() {
            continue;
        }

        // Standard hash-based intersection logic
        for &v in &adj1[u] {
            hash[v] = true;
        }

        for &v in &adj0[u] {
            if u < v {
                for &w in &adj1[v] {
                    if hash[w] {
                        count += 1;
                    }
                }
            }
        }

        // Clean up hash for the next iteration
        for &v in &adj1[u] {
            hash[v] = false;
        }
    }

    count
}

// reexport_module_members!("rust_core.triangle.cetc" from "rust_core._core.triangle.cetc");
