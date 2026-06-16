use crate::graph::AdjList;
use std::collections::VecDeque;

/// A simple BFS to calculate levels/distances from a starting component.
/// In most CETC contexts, this assumes node 0 is the root or
/// it iterates through all components.
pub fn bfs<W>(adj: &AdjList<W>) -> Vec<i32> {
    let n = adj.n();
    let mut levels = vec![-1; n];
    let mut queue = VecDeque::new();

    for i in 0..n {
        if levels[i] == -1 {
            levels[i] = 0;
            queue.push_back(i);
            while let Some(u) = queue.pop_front() {
                for &(v_node, ref _w) in &adj.adj[u] {
                    let v = v_node as usize;
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
