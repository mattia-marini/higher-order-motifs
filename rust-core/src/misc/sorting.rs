use crate::graph::AdjList;
use pyo3::prelude::PyResult;

/// Returns a degree ordering of the vertices, the position of each vertex in that ordering, and
/// the maximum degree of the graph.
/// Time Complexity: O(n)
pub fn degree_ordering(adj: &AdjList, decreasing: bool) -> (Vec<usize>, Vec<usize>, usize) {
    let n = adj.n();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0);
    }

    let deg: Vec<usize> = adj.adj.iter().map(|neighbors| neighbors.len()).collect();
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

    if decreasing {
        for v in (0..n).rev() {
            let d = deg[v];
            pos[v] = bin_starts[d];
            order[pos[v]] = v;
            bin_starts[d] += 1;
        }
    } else {
        for v in 0..n {
            let d = deg[v];
            pos[v] = bin_starts[d];
            order[pos[v]] = v;
            bin_starts[d] += 1;
        }
    }

    (order, pos, max_deg)
}

/// Efficiently sorts each adjacency list in-place.
/// Time Complexity: O(n + m)
/// Space Complexity: O(n + m)
pub fn sort_adj_list(adj: &AdjList) -> Vec<Vec<usize>> {
    let n = adj.n();
    let mut rv = vec![Vec::new(); n];

    for u in 0..n {
        rv[u].reserve(adj.adj[u].len());
    }

    for u in 0..n {
        for &v in &adj.adj[u] {
            rv[v as usize].push(u);
        }
    }
    rv
}

/// Returns a degeneracy ordering of the graph, the position of each vertex,
/// and the degeneracy (k) of the graph.
///
/// Complexity: O(n + m)
pub fn degeneracy_ordering(adj: &AdjList) -> (Vec<usize>, Vec<usize>, usize) {
    let n = adj.n();
    if n == 0 {
        return (vec![], vec![], 0);
    }

    // 1. Calculate degrees and find max degree
    let mut deg: Vec<usize> = adj.adj.iter().map(|neighbors| neighbors.len()).collect();
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

        for &u_node in &adj.adj[v] {
            let u = u_node as usize;
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
pub fn degeneracy_ordering_py(adj: &AdjList) -> PyResult<(Vec<usize>, Vec<usize>, usize)> {
    let n = adj.n();
    if n == 0 {
        return Ok((vec![], vec![], 0));
    }

    let mut deg: Vec<usize> = adj.adj.iter().map(|neighbors| neighbors.len()).collect();
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
        for &u_node in &adj.adj[v] {
            let u = u_node as usize;
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
