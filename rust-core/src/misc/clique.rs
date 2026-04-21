use crate::{graph::AdjList, misc::neighbors_sorted_list_cloj};

use super::degree_ordering;

// pub fn clique_3_cloj<F>(adj: &AdjList, sort_degrees: bool, mut cloj: F)
// where
//     F: FnMut(usize, usize, usize),
// {
//     let n = adj.n();
//     let mut a = vec![Vec::new(); n];
//     for i in 0..n {
//         a[i].reserve(adj[i].len());
//     }
//
//     if sort_degrees {
//         let (order, pos, _) = degree_ordering(adj, true);
//
//         for i in 0..n {
//             let u = order[i];
//             for &v in &adj[u] {
//                 // Only process directed edge u -> v in the DAG
//                 if i < pos[v as usize] {
//                     // Count common neighbors in the 'a' sets
//                     neighbors_sorted_list_cloj(&a[u], &a[v as usize], |w| cloj(u, v, w));
//                     a[v as usize].push(pos[u]);
//                 }
//             }
//         }
//     } else {
//         for u in 0..n {
//             for &v in &adj[u] {
//                 if u < v {
//                     neighbors_sorted_list_cloj(&a[u], &a[v], |w| cloj(u, v, w));
//                     a[v].push(u);
//                 }
//             }
//         }
//     }
// }
//
// pub fn clique_4_cloj<F>(adj: &AdjList, sort_degrees: bool, mut cloj: F)
// where
//     F: FnMut(usize, usize, usize, usize),
// {
//     let n = adj.len();
//     let mut a = vec![Vec::new(); n];
//     for i in 0..n {
//         a[i].reserve(adj[i].len());
//     }
//
//     if sort_degrees {
//         let (order, pos, _) = degree_ordering(adj, true);
//
//         for i in 0..n {
//             let u = order[i];
//             for &v in &adj[u] {
//                 // Only process directed edge u -> v in the DAG
//                 if i < pos[v] {
//                     // Count common neighbors in the 'a' sets
//                     let mut common_neighbors = Vec::new();
//                     neighbors_sorted_list_cloj(&a[u], &a[v], |w| common_neighbors.push(w));
//                     for &w in &common_neighbors {
//                         // For each common neighbor w, find common neighbors of w and v
//                         neighbors_sorted_list_cloj(&a[w], &common_neighbors, |x| cloj(u, v, w, x));
//                     }
//
//                     a[v].push(pos[u]);
//                 }
//             }
//         }
//     } else {
//         for u in 0..n {
//             for &v in &adj[u] {
//                 if u < v {
//                     let mut common_neighbors = Vec::new();
//                     neighbors_sorted_list_cloj(&a[u], &a[v], |w| common_neighbors.push(w));
//                     for &w in &common_neighbors {
//                         // For each common neighbor w, find common neighbors of w and v
//                         neighbors_sorted_list_cloj(&a[w], &common_neighbors, |x| cloj(u, v, w, x));
//                     }
//                     a[v].push(u);
//                 }
//             }
//         }
//     }
// }
