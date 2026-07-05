use crate::types::NodeId;

/// Efficiently computes the common elements shared by two sorted lists.
/// Time Complexity: O(deg(u) + deg(v))
pub fn common_neighbors_sorted_list<'a, T: Eq + Ord>(a: &'a [T], b: &'a [T]) -> Vec<&'a T> {
    let mut neighbors = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            neighbors.push(&a[i]);
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

/// Efficiently computes the common elements shared by three sorted lists.
/// Time Complexity: O(deg(u) + deg(v) + deg(w))
pub fn common_neighbors_sorted_list_3<'a, T: Eq + Ord>(
    a: &'a [T],
    b: &'a [T],
    c: &'a [T],
    hint: T,
) -> Vec<&'a T> {
    let mut neighbors = Vec::new();
    let (mut i, mut j, mut k) = (0, 0, 0);

    while i < a.len() && a[i] < hint && j < b.len() && b[j] < hint && k < c.len() && c[k] < hint {
        if a[i] == b[j] && b[j] == c[k] {
            neighbors.push(&a[i]);
            i += 1;
            j += 1;
            k += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else if b[j] < c[k] {
            j += 1;
        } else {
            k += 1;
        }
    }
    neighbors
}

/// Counts the number of common neighbors.
pub fn count_common_neighbors_sorted_list<T: Eq + Ord>(a: &[T], b: &[T]) -> usize {
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

/// Efficiently computes the common elements shared by three sorted lists.
/// Time Complexity: O(deg(u) + deg(v) + deg(w))
pub fn count_common_neighbors_sorted_list_3<T: Eq + Ord>(
    a: &[T],
    b: &[T],
    c: &[T],
    hint: T,
) -> usize {
    let mut count = 0;
    let (mut i, mut j, mut k) = (0, 0, 0);

    while i < a.len() && a[i] < hint && j < b.len() && b[j] < hint && k < c.len() && c[k] < hint {
        if a[i] == b[j] && b[j] == c[k] {
            count += 1;
            i += 1;
            j += 1;
            k += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else if b[j] < c[k] {
            j += 1;
        } else {
            k += 1;
        }
    }
    count
}

/// Executes a closure for each common neighbor found.
pub fn neighbors_sorted_list_cloj<T: Eq + Ord, F>(a: &[T], b: &[T], mut f: F)
where
    F: FnMut(&T),
{
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            f(&a[i]);
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
}

/// Efficiently computes the common elements shared by three sorted lists.
/// Time Complexity: O(deg(u) + deg(v) + deg(w))
pub fn common_neighbors_sorted_list_3_cloj<T: Eq + Ord, F>(
    a: &[T],
    b: &[T],
    c: &[T],
    hint: T,
    mut f: F,
) where
    F: FnMut(usize, usize, usize),
{
    let (mut i, mut j, mut k) = (0, 0, 0);

    while i < a.len() && a[i] < hint && j < b.len() && b[j] < hint && k < c.len() && c[k] < hint {
        if a[i] == b[j] && b[j] == c[k] {
            f(i, j, k);
            i += 1;
            j += 1;
            k += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else if b[j] < c[k] {
            j += 1;
        } else {
            k += 1;
        }
    }
}
