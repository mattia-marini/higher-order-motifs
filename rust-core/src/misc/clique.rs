use super::{count_neighbors_sorted_list, degree_ordering};

pub fn clique_3(adj: &Vec<Vec<usize>>, sort_degrees: bool) -> usize {
    let n = adj.len();
    let mut a = vec![Vec::new(); n];
    for i in 0..n {
        a[i].reserve(adj[i].len());
    }

    let mut count = 0;

    if sort_degrees {
        let (order, pos, _) = degree_ordering(adj, true);

        for i in 0..n {
            let u = order[i];
            for &v in &adj[u] {
                // Only process directed edge u -> v in the DAG
                if i < pos[v] {
                    // Count common neighbors in the 'a' sets
                    count += count_neighbors_sorted_list(&a[u], &a[v]);
                    a[v].push(pos[u]);
                }
            }
        }
    } else {
        for u in 0..n {
            for &v in &adj[u] {
                if u < v {
                    count += count_neighbors_sorted_list(&a[u], &a[v]);
                    a[v].push(u);
                }
            }
        }
    }
    count
}

pub fn clique_4(adj: &Vec<Vec<usize>>, sort_degrees: bool) -> usize {
    let n = adj.len();
    let mut a = vec![Vec::new(); n];
    for i in 0..n {
        a[i].reserve(adj[i].len());
    }

    let mut count = 0;

    if sort_degrees {
        let (order, pos, _) = degree_ordering(adj, true);

        for i in 0..n {
            let u = order[i];
            for &v in &adj[u] {
                // Only process directed edge u -> v in the DAG
                if i < pos[v] {
                    // Count common neighbors in the 'a' sets
                    count += count_neighbors_sorted_list(&a[u], &a[v]);
                    a[v].push(pos[u]);
                }
            }
        }
    } else {
        for u in 0..n {
            for &v in &adj[u] {
                if u < v {
                    count += count_neighbors_sorted_list(&a[u], &a[v]);
                    a[v].push(u);
                }
            }
        }
    }
    count
}
