use crate::graph::types::NodeId;

/// Efficiently computes the common neighbors of two vertices given their sorted adjacency lists.
/// Time Complexity: O(deg(u) + deg(v))
pub fn common_neighbors_sorted_list(a: &[NodeId], b: &[NodeId]) -> Vec<NodeId> {
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

pub fn count_neighbors_sorted_list(a: &[NodeId], b: &[NodeId]) -> usize {
    let mut count = 0;
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            count += 1;
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    count
}

pub fn neighbors_sorted_list_cloj<F>(a: &[NodeId], b: &[NodeId], mut f: F)
where
    F: FnMut(NodeId),
{
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            f(a[i]);
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
}
