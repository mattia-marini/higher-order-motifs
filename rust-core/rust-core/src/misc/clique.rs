use hashbrown::{HashMap, HashSet};
use pyo3::pymethods;
use pyo3_stub_gen::derive::{gen_stub_pyfunction, gen_stub_pymethods};
use std::hash::Hash;

use crate::{
    graph::{AdjList, NodeId},
    misc::neighbors_sorted_list_cloj,
};

// use super::degree_ordering;
// pub fn clique_3_cloj<F>(adj: &AdjList, sort_degrees: bool, mut cloj: F)
// where
//     F: FnMut(NodeId, NodeId, NodeId),
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
//             let u = order[i] as usize;
//             for &v in &adj[u as usize] {
//                 let v = v as usize;
//                 if i < pos[v] as usize {
//                     neighbors_sorted_list_cloj(&a[u], &a[v], |w| {
//                         cloj(u as NodeId, v as NodeId, w as NodeId)
//                     });
//                     a[v].push(pos[u]);
//                 }
//             }
//         }
//     } else {
//         for u in 0..n {
//             for &v in &adj[u] {
//                 let v = v as usize;
//                 if u < v {
//                     neighbors_sorted_list_cloj(&a[u], &a[v], |w| {
//                         cloj(u as NodeId, v as NodeId, w as NodeId)
//                     });
//                     a[v].push(u as NodeId);
//                 }
//             }
//         }
//     }
// }
//
// pub fn enum_clique_4(adj: &AdjList, sort_degrees: bool) -> (Vec<H2>, Vec<H3>, Vec<H4>) {
//     let n = adj.n();
//     let mut a = vec![Vec::new(); n];
//     for i in 0..n {
//         a[i].reserve(adj[i].len());
//     }
//
//     let mut clique_2: Vec<H2> = Vec::new();
//     let mut clique_3: Vec<H3> = Vec::new();
//     let mut clique_4: Vec<H4> = Vec::new();
//
//     if sort_degrees {
//         let (order, pos, _) = degree_ordering(adj, true);
//
//         for i in 0..n {
//             let u = order[i] as usize;
//             for &v in &adj[u] {
//                 let v = v as usize;
//                 if i < pos[v] as usize {
//                     let mut common_neighbors = Vec::new();
//                     neighbors_sorted_list_cloj(&a[u], &a[v], |w| common_neighbors.push(w));
//                     if common_neighbors.is_empty() {
//                         clique_2.push(H2::new(u as NodeId, v as NodeId).unwrap());
//                         a[v].push(pos[u as usize]);
//                         continue;
//                     }
//
//                     let mut clique_4_vertexes = HashSet::new();
//                     for &w in &common_neighbors {
//                         neighbors_sorted_list_cloj(&a[w as usize], &common_neighbors, |x| {
//                             clique_4_vertexes.insert(x);
//                         });
//                     }
//
//                     for &w in &common_neighbors {
//                         if clique_4_vertexes.contains(&w) {
//                             clique_4.push(
//                                 H4::new(u as NodeId, v as NodeId, w as NodeId, w as NodeId)
//                                     .unwrap(),
//                             );
//                         } else {
//                             clique_3.push(H3::new(u as NodeId, v as NodeId, w as NodeId).unwrap());
//                         }
//                     }
//
//                     a[v].push(pos[u as usize]);
//                 }
//             }
//         }
//     } else {
//         for u in 0..n {
//             for &v in &adj[u] {
//                 let v = v as usize;
//                 if u < v {
//                     let mut common_neighbors = Vec::new();
//                     neighbors_sorted_list_cloj(&a[u], &a[v], |w| common_neighbors.push(w));
//                     if common_neighbors.is_empty() {
//                         clique_2.push(H2::new(u as NodeId, v as NodeId).unwrap());
//                         a[v].push(u as NodeId);
//                         continue;
//                     }
//
//                     let mut clique_4_vertexes = HashSet::new();
//                     let mut clique_4_edges = HashSet::new();
//                     for &w in &common_neighbors {
//                         neighbors_sorted_list_cloj(&a[w as usize], &common_neighbors, |x| {
//                             let edge = if w < x { (w, x) } else { (x, w) };
//                             clique_4_vertexes.insert(x);
//                             clique_4_edges.insert(edge);
//                             // clique_4.push((u as NodeId, v as NodeId, w as NodeId, x as NodeId));
//                         });
//                     }
//
//                     for (w, x) in clique_4_edges {
//                         clique_4.push(H4::new(u as NodeId, v as NodeId, w, x).uwnrap());
//                     }
//
//                     for &w in &common_neighbors {
//                         if !clique_4_vertexes.contains(&w) {
//                             clique_3.push(H3::new(u as NodeId, v as NodeId, w as NodeId).unwrap());
//                         }
//                     }
//
//                     a[v].push(u as NodeId);
//                 }
//             }
//         }
//     }
//     (clique_2, clique_3, clique_4)
// }

#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl AdjList {
    /// Returns all maximal cliques in the undirected graph.
    ///
    /// This is an iterative implementation of the Bron-Kerbosch algorithm with pivoting,
    /// translating the provided Python implementation to avoid recursion depth limits.
    pub fn find_cliques(&self) -> Vec<Vec<NodeId>> {
        let mut cliques = Vec::new();

        if self.n() == 0 {
            return cliques;
        }

        // Convert adjacency list to HashSets for O(1) lookups and set operations (ignores self-loops)
        let mut adj_sets: Vec<HashSet<NodeId>> = Vec::with_capacity(self.n());
        for u in 0..self.n() {
            let mut set = HashSet::new();
            for &v in &self.adj[u] {
                if v as usize != u {
                    set.insert(v);
                }
            }
            adj_sets.push(set);
        }

        // Initialize candidate sets
        let mut cand: HashSet<NodeId> = (0..self.n() as NodeId).collect();
        let mut subg: HashSet<NodeId> = cand.clone();

        if cand.is_empty() {
            return cliques;
        }

        let mut stack = Vec::new();
        let mut q: Vec<NodeId> = Vec::new();

        // Placeholder for Q[-1] logic in Python
        q.push(0);

        // Find initial pivot
        let u_pivot = *subg
            .iter()
            .max_by_key(|&&u| cand.intersection(&adj_sets[u as usize]).count())
            .unwrap();

        let mut ext_u: Vec<NodeId> = cand
            .difference(&adj_sets[u_pivot as usize])
            .cloned()
            .collect();

        loop {
            if let Some(q_node) = ext_u.pop() {
                cand.remove(&q_node);

                // Q[-1] = q
                if let Some(last) = q.last_mut() {
                    *last = q_node;
                }

                let adj_q = &adj_sets[q_node as usize];
                let subg_q: HashSet<NodeId> = subg.intersection(adj_q).cloned().collect();

                if subg_q.is_empty() {
                    // Yield Q[:]
                    cliques.push(q.clone());
                } else {
                    let cand_q: HashSet<NodeId> = cand.intersection(adj_q).cloned().collect();
                    if !cand_q.is_empty() {
                        // Push state to stack
                        stack.push((subg, cand, ext_u));
                        q.push(0); // Q.append(None)

                        subg = subg_q;
                        cand = cand_q;

                        // Find new pivot
                        let u_pivot = *subg
                            .iter()
                            .max_by_key(|&&u| cand.intersection(&adj_sets[u as usize]).count())
                            .unwrap();

                        ext_u = cand
                            .difference(&adj_sets[u_pivot as usize])
                            .cloned()
                            .collect();
                    }
                }
            } else {
                // Backtrack
                q.pop();
                if let Some((prev_subg, prev_cand, prev_ext)) = stack.pop() {
                    subg = prev_subg;
                    cand = prev_cand;
                    ext_u = prev_ext;
                } else {
                    break;
                }
            }
        }

        cliques
    }
}
