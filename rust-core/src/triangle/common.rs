use std::collections::VecDeque;

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_stub_gen::reexport_module_members;

#[pymodule(submodule)]
pub mod common {
    use pyo3::prelude::*;
    use pyo3::types::PyList;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.common")]
    pub fn degree_ordering(adj: Vec<Vec<usize>>) -> (Vec<usize>, Vec<usize>, usize) {
        super::degree_ordering(&adj)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.common")]
    pub fn bfs(adj: Vec<Vec<usize>>) -> Vec<i32> {
        super::bfs(&adj)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.common")]
    pub fn sort_adj_list(adj: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        let mut adj_mut = adj.clone();
        super::sort_adj_list(&mut adj_mut);
        adj_mut
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.common")]
    pub fn common_neighbors_sorted_list(a: Vec<usize>, b: Vec<usize>) -> Vec<usize> {
        super::common_neighbors_sorted_list(&a, &b)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.common")]
    pub fn degeneracy_ordering(adj: Vec<Vec<usize>>) -> (Vec<usize>, Vec<usize>, usize) {
        super::degeneracy_ordering(&adj)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.triangle.common")]
    pub fn degeneracy_ordering_py(
        adj: Bound<'_, PyList>,
    ) -> PyResult<(Vec<usize>, Vec<usize>, usize)> {
        super::degeneracy_ordering_py(adj)
    }
}

/// Returns a degree ordering of the vertices, the position of each vertex in that ordering, and
/// the maximum degree of the graph.
/// Time Complexity: O(n + m)
pub fn degree_ordering(adj: &Vec<Vec<usize>>) -> (Vec<usize>, Vec<usize>, usize) {
    let now = std::time::Instant::now();

    let n = adj.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0);
    }

    let deg: Vec<usize> = adj.iter().map(|neighbors| neighbors.len()).collect();
    let max_deg = *deg.iter().max().unwrap_or(&0);

    // Count how many vertices have each degree
    let mut bin_count = vec![0usize; max_deg + 1];
    for &d in &deg {
        bin_count[d] += 1;
    }

    // Compute starting index for each degree bin
    let mut start_pos = 0usize;
    let mut bin_starts = vec![0usize; max_deg + 1];
    for d in 0..=max_deg {
        bin_starts[d] = start_pos;
        start_pos += bin_count[d];
    }

    // Fill order and pos
    let mut order = vec![0usize; n];
    let mut pos = vec![0usize; n];
    for v in 0..n {
        let d = deg[v];
        pos[v] = bin_starts[d];
        order[pos[v]] = v;
        bin_starts[d] += 1;
    }

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    (order, pos, max_deg)
}

/// A simple BFS to calculate levels/distances from a starting component.
/// In most CETC contexts, this assumes node 0 is the root or
/// it iterates through all components.
pub fn bfs(adj: &Vec<Vec<usize>>) -> Vec<i32> {
    let n = adj.len();
    let mut levels = vec![-1; n];
    let mut queue = VecDeque::new();

    for i in 0..n {
        if levels[i] == -1 {
            levels[i] = 0;
            queue.push_back(i);
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if levels[v] == -1 {
                        levels[v] = levels[u] + 1;
                        queue.push_back(v);
                    }
                }
            }
        }
    }
    levels
}

/// Efficiently computes the common neighbors of two vertices given their sorted adjacency lists.
/// Time Complexity: O(deg(u) + deg(v))
pub fn common_neighbors_sorted_list(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut neighbors = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            neighbors.push(a[i]);
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    neighbors
}

/// Efficiently sorts each adjacency list in-place.
/// Time Complexity: O(n + m)
/// Space Complexity: O(n + m)
pub fn sort_adj_list(adj: &mut Vec<Vec<usize>>) {
    let n = adj.len();

    // 1. Calculate degrees (counts) for each vertex
    let mut degrees = vec![0; n];
    for neighbors in adj.iter() {
        for &v in neighbors {
            degrees[v] += 1;
        }
    }

    // 2. Create the flat buffer and "starting positions" (offsets)
    // We are essentially building a CSR (Compressed Sparse Row) structure
    let mut flat_sorted = vec![0; adj.iter().map(|x| x.len()).sum()];
    let mut offsets = vec![0; n + 1];
    for i in 0..n {
        offsets[i + 1] = offsets[i] + degrees[i];
    }

    // 3. Current insertion pointers for each vertex's bucket
    let mut current_pos = offsets.clone();

    // 4. Populate the flat buffer
    // By iterating through vertices and placing them in their neighbors' buckets,
    // we naturally result in sorted lists for the neighbors.
    for u in 0..n {
        // Note: we can't drain 'adj' easily while holding 'u', so we iterate
        for i in 0..adj[u].len() {
            let v = adj[u][i];
            let dest = current_pos[v];
            flat_sorted[dest] = u;
            current_pos[v] += 1;
        }
    }

    // 5. Redistribute the flat buffer back into adj
    for i in 0..n {
        let start = offsets[i];
        let end = offsets[i + 1];
        // Efficiently replace the inner vector content
        adj[i].clear();
        adj[i].extend_from_slice(&flat_sorted[start..end]);
    }
}

/// Returns a degeneracy ordering of the graph, the position of each vertex,
/// and the degeneracy (k) of the graph.
///
/// Complexity: O(n + m)
pub fn degeneracy_ordering(adj: &[Vec<usize>]) -> (Vec<usize>, Vec<usize>, usize) {
    let n = adj.len();
    if n == 0 {
        return (vec![], vec![], 0);
    }

    // 1. Calculate degrees and find max degree
    let mut deg: Vec<usize> = adj.iter().map(|neighbors| neighbors.len()).collect();
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

    // 5. Main loop: remove node of minimum degree
    let mut k = 0;
    for i in 0..n {
        let v = order[i];
        k = std::cmp::max(k, deg[v]);

        for &u in &adj[v] {
            if pos[u] > i {
                // Only look at neighbors still "in the graph"
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

/// A version of degeneracy_ordering that accepts Python objects.
/// It maps Python objects to internal indices to perform the O(n + m) sort.
pub fn degeneracy_ordering_py(
    adj_py: Bound<'_, PyList>,
) -> PyResult<(Vec<usize>, Vec<usize>, usize)> {
    let n = adj_py.len();
    if n == 0 {
        return Ok((vec![], vec![], 0));
    }

    let mut deg: Vec<usize> = adj_py
        .iter()
        .map(|neighbors| neighbors.len().unwrap())
        .collect();
    let max_deg = *deg.iter().max().unwrap_or(&0);

    let mut bin_count = vec![0; max_deg + 1];
    for &d in &deg {
        bin_count[d] += 1;
    }

    let mut bin_starts = vec![0; max_deg + 1];
    let mut start_pos = 0;
    for d in 0..=max_deg {
        bin_starts[d] = start_pos;
        start_pos += bin_count[d];
    }

    let mut temp_starts = bin_starts.clone();
    let mut order_idx = vec![0; n];
    let mut pos = vec![0; n];
    for v in 0..n {
        pos[v] = temp_starts[deg[v]];
        order_idx[pos[v]] = v;
        temp_starts[deg[v]] += 1;
    }

    let mut k = 0;
    for i in 0..n {
        let v = order_idx[i];
        k = std::cmp::max(k, deg[v]);
        let neigboors = adj_py.get_item(v).unwrap().cast_into::<PyList>().unwrap();
        for u in neigboors.iter() {
            let u = u.extract::<usize>().unwrap();
            if pos[u] > i {
                let u_deg = deg[u];
                let u_pos = pos[u];
                let first_node_pos = bin_starts[u_deg];
                let first_node = order_idx[first_node_pos];

                if u != first_node {
                    pos.swap(u, first_node);
                    order_idx.swap(u_pos, first_node_pos);
                }

                bin_starts[u_deg] += 1;
                deg[u] -= 1;
            }
        }
    }

    Ok((order_idx, pos, k))
}

reexport_module_members!("rust_core.triangle.common" from "rust_core.core.triangle.common");
