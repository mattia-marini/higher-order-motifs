use crate::graph::hyperedge::NodeId;

/// Efficiently computes the common elements shared by two sorted lists.
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

/// Efficiently computes the common elements shared by three sorted lists.
/// Time Complexity: O(deg(u) + deg(v))
pub fn common_neighbors_sorted_list_3<'a, W>(
    a: &'a [(NodeId, W)],
    b: &'a [(NodeId, W)],
    c: &'a [(NodeId, W)],
    hint: NodeId,
) -> Vec<&'a (NodeId, W)> {
    let mut neighbors = Vec::new();
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < a.len()
        && a[i].0 < hint
        && j < b.len()
        && b[j].0 < hint
        && k < c.len()
        && c[k].0 < hint
    {
        if a[i].0 == b[j].0 && b[j].0 == c[k].0 {
            neighbors.push(&a[i]);
            i += 1;
            j += 1;
            k += 1;
        } else {
            let min_id = a[i].0.min(b[j].0).min(c[k].0);
            i += (a[i].0 == min_id) as usize;
            j += (b[j].0 == min_id) as usize;
            k += (c[k].0 == min_id) as usize;
        }
    }
    neighbors
}

/// Counts the number of common neighbors.
pub fn count_common_neighbors_sorted_list<W>(a: &[(NodeId, W)], b: &[(NodeId, W)]) -> usize {
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

/// Efficiently computes the common elements shared by three sorted lists.
/// Time Complexity: O(deg(u) + deg(v))
pub fn count_common_neighbors_sorted_list_3<'a, W>(
    a: &'a [(NodeId, W)],
    b: &'a [(NodeId, W)],
    c: &'a [(NodeId, W)],
    hint: NodeId,
) -> usize {
    let mut count = 0;
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < a.len()
        && a[i].0 < hint
        && j < b.len()
        && b[j].0 < hint
        && k < c.len()
        && c[k].0 < hint
    {
        if a[i].0 == b[j].0 && b[j].0 == c[k].0 {
            count += 1;
            i += 1;
            j += 1;
            k += 1;
        } else {
            let min_id = a[i].0.min(b[j].0).min(c[k].0);
            i += (a[i].0 == min_id) as usize;
            j += (b[j].0 == min_id) as usize;
            k += (c[k].0 == min_id) as usize;
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

/// Efficiently computes the common elements shared by three sorted lists.
/// Time Complexity: O(deg(u) + deg(v))
pub fn common_neighbors_sorted_list_3_cloj<'a, W, F>(
    a: &'a [(NodeId, W)],
    b: &'a [(NodeId, W)],
    c: &'a [(NodeId, W)],
    hint: NodeId,
    mut f: F,
) where
    F: FnMut(usize, usize, usize),
{
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < a.len()
        && a[i].0 < hint
        && j < b.len()
        && b[j].0 < hint
        && k < c.len()
        && c[k].0 < hint
    {
        if a[i].0 == b[j].0 && b[j].0 == c[k].0 {
            f(i, j, k);
            i += 1;
            j += 1;
            k += 1;
        } else {
            let min_id = a[i].0.min(b[j].0).min(c[k].0);
            i += (a[i].0 == min_id) as usize;
            j += (b[j].0 == min_id) as usize;
            k += (c[k].0 == min_id) as usize;
        }
    }
}
