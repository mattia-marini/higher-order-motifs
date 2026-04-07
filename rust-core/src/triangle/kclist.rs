use pyo3::{prelude::*, types::PyList};
use pyo3_stub_gen::reexport_module_members;

#[pymodule(submodule)]
pub mod kclist {
    use pyo3::{Bound, PyResult, pyfunction, types::PyList};
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.kclist")]
    pub fn kclist(adj: Vec<Vec<usize>>) -> usize {
        super::kclist(&adj)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.kclist")]
    pub fn kclist_py(adj: Bound<'_, PyList>) -> PyResult<usize> {
        super::kclist_py(adj)
    }
}

/// Counts the number of triangles in the graph using the Chiba-Nishizeki algorithm.
///
/// Complexity: O(m * k) where k is the degeneracy.
pub fn kclist(adj: &[Vec<usize>]) -> usize {
    let n = adj.len();
    if n < 3 {
        return 0;
    }

    // 1. Get the degeneracy ordering
    let (order, pos, _) = super::common::degeneracy_ordering(adj);

    // 2. Re-orient edges: only keep edges u -> v where pos[u] < pos[v]
    // This creates a Directed Acyclic Graph (DAG)
    let mut out_adj: Vec<Vec<usize>> = vec![vec![]; n];
    for u in 0..n {
        for &v in &adj[u] {
            if pos[u] < pos[v] {
                out_adj[u].push(v);
            }
        }
    }

    // 3. Triangle counting
    let mut count = 0;
    // We use usize::MAX as the "unmarked" value since 0 is a valid vertex ID
    let mut marks = vec![usize::MAX; n];

    for &u in &order {
        // Mark all neighbors of u
        for &v in &out_adj[u] {
            marks[v] = u;
        }

        // Check neighbors of neighbors
        for &v in &out_adj[u] {
            for &w in &out_adj[v] {
                if marks[w] == u {
                    count += 1;
                }
            }
        }
    }

    count
}

/// A version of kclist that accepts Python objects.
pub fn kclist_py(adj_py: Bound<'_, PyList>) -> PyResult<usize> {
    let n = adj_py.len();
    if n < 3 {
        return Ok(0);
    }

    // 1. Get the degeneracy ordering using your conversion
    // We call it directly as a Rust function
    let (order, pos, _) = super::common::degeneracy_ordering_py(adj_py.clone())?;

    // 2. Re-orient edges: only keep edges u -> v where pos[u] < pos[v]
    // We'll build this in Rust memory to ensure the triangle counting is fast.
    let mut out_adj: Vec<Vec<usize>> = vec![vec![]; n];

    for u in 0..n {
        let neighbors = adj_py.get_item(u)?.cast_into::<PyList>()?;
        for v_any in neighbors.iter() {
            let v = v_any.extract::<usize>()?;
            if pos[u] < pos[v] {
                out_adj[u].push(v);
            }
        }
    }

    // 3. Triangle counting
    let mut count = 0;
    let mut marks = vec![usize::MAX; n];

    for &u in &order {
        // Mark all "out-neighbors" of u
        for &v in &out_adj[u] {
            marks[v] = u;
        }

        // Check neighbors of neighbors
        for &v in &out_adj[u] {
            for &w in &out_adj[v] {
                if marks[w] == u {
                    count += 1;
                }
            }
        }
    }

    Ok(count)
}

reexport_module_members!("rust_core.triangle.kclist" from "rust_core.core.triangle.kclist");
