use crate::graph::types::NodeId;

/// Efficiently computes the common neighbors of two vertices given their sorted adjacency lists.
/// Time Complexity: O(deg(u) + deg(v))
pub fn common_neighbors_sorted_list<'a, W>(
    a: &'a [(NodeId, W)],
    b: &'a [(NodeId, W)],
) -> Vec<&'a (NodeId, W)> {
    let mut neighbors = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].0 == b[j].0 {
            neighbors.push(&a[i]);
            i += 1;
            j += 1;
        } else if a[i].0 < b[j].0 {
            i += 1;
        } else {
            j += 1;
        }
    }
    neighbors
}

/// Counts the number of common neighbors.
pub fn count_neighbors_sorted_list<W>(a: &[(NodeId, W)], b: &[(NodeId, W)]) -> usize {
    let mut count = 0;
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].0 == b[j].0 {
            count += 1;
            i += 1;
            j += 1;
        } else if a[i].0 < b[j].0 {
            i += 1;
        } else {
            j += 1;
        }
    }
    count
}

/// Executes a closure for each common neighbor found.
pub fn neighbors_sorted_list_cloj<W, F>(a: &[(NodeId, W)], b: &[(NodeId, W)], mut f: F)
where
    F: FnMut(NodeId),
{
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].0 == b[j].0 {
            f(a[i].0);
            i += 1;
            j += 1;
        } else if a[i].0 < b[j].0 {
            i += 1;
        } else {
            j += 1;
        }
    }
}
