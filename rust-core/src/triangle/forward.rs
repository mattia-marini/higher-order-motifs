use super::common::degree_ordering;
use pyo3::prelude::*;
use std::collections::HashSet;

/// Computes the degree ordering: (order, position)

#[pymodule(submodule)]
pub mod forward {
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.triangle.forward")]
    pub fn forward(adj: Vec<Vec<usize>>, sort_degrees: bool) -> usize {
        super::forward(&adj, sort_degrees)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.triangle.forward")]
    pub fn compact_forward(adj: Vec<Vec<usize>>, sort_degrees: bool) -> usize {
        super::forward(&adj, sort_degrees)
    }
}

pub fn forward(adj: &Vec<Vec<usize>>, sort_degrees: bool) -> usize {
    let n = adj.len();
    let mut a: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut count = 0;

    if sort_degrees {
        let pos = degree_ordering(adj).1;
        for u in 0..n {
            for &v in &adj[u] {
                // Only process directed edge u -> v in the DAG
                if pos[u] < pos[v] {
                    // Count common neighbors in the 'a' sets
                    count += a[u].intersection(&a[v]).count();
                    a[v].insert(u);
                }
            }
        }
    } else {
        for u in 0..n {
            for &v in &adj[u] {
                if u < v {
                    count += a[u].intersection(&a[v]).count();
                    a[v].insert(u);
                }
            }
        }
    }
    count
}

pub fn compact_forward(adj: &Vec<Vec<usize>>, sort_degrees: bool) -> usize {
    let n = adj.len();
    let mut a: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut hash_map = vec![false; n];
    let mut count = 0;

    let pos = if sort_degrees {
        degree_ordering(adj).1
    } else {
        vec![]
    };

    for u in 0..n {
        for &v in &adj[u] {
            let is_ordered = if sort_degrees { pos[u] < pos[v] } else { u < v };

            if is_ordered {
                // Mark elements in a[u]
                for &w in &a[u] {
                    hash_map[w] = true;
                }
                // Check elements in a[v] against the mark
                for &w in &a[v] {
                    if hash_map[w] {
                        count += 1;
                    }
                }
                // Reset marks for next iteration
                for &w in &a[u] {
                    hash_map[w] = false;
                }
                a[v].insert(u);
            }
        }
    }
    count
}

// reexport_module_members!("rust_core.triangle.forward" from "rust_core._core.triangle.forward");
