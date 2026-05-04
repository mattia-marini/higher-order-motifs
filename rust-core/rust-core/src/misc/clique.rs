use std::collections::HashSet;

use crate::{
    graph::{
        AdjList,
        types::{H2, H3, H4, NodeId},
    },
    misc::neighbors_sorted_list_cloj,
};

use super::degree_ordering;

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

impl AdjList {
    /// Enumerate all maximal cliques
    /// Returns a vector of maximal cliques, each clique as Vec<NodeId>.
    /// Node order inside cliques is deterministic by node id.
    pub fn enum_cliques(&self) -> Vec<Vec<NodeId>> {
        let n = self.adj.len();
        if n == 0 {
            return Vec::new();
        }

        // Build adjacency bitsets for fast set ops.
        // neighbors_bits[v] has bit u set iff (v,u) is an edge.
        let words = (n + 63) / 64;
        let mut neighbors_bits = vec![vec![0u64; words]; n];

        for v in 0..n {
            for &u_id in &self.adj[v] {
                let u = u_id as usize;
                if u < n && u != v {
                    neighbors_bits[v][u >> 6] |= 1u64 << (u & 63);
                }
            }
        }

        // Initial P = all vertices, X = empty, R = empty
        let mut p0 = vec![0u64; words];
        for v in 0..n {
            p0[v >> 6] |= 1u64 << (v & 63);
        }
        let x0 = vec![0u64; words];
        let r0: Vec<NodeId> = Vec::new();

        #[derive(Clone)]
        struct Frame {
            r: Vec<NodeId>,
            p: Vec<u64>,
            x: Vec<u64>,
            // Candidates = P \ N(pivot), materialized once per frame
            candidates: Vec<usize>,
            next_idx: usize,
        }

        fn bitset_is_empty(bs: &[u64]) -> bool {
            bs.iter().all(|&w| w == 0)
        }

        fn bitset_and(a: &[u64], b: &[u64]) -> Vec<u64> {
            a.iter().zip(b).map(|(&x, &y)| x & y).collect()
        }

        fn bitset_andnot(a: &[u64], b: &[u64]) -> Vec<u64> {
            a.iter().zip(b).map(|(&x, &y)| x & !y).collect()
        }

        fn bitset_or(a: &[u64], b: &[u64]) -> Vec<u64> {
            a.iter().zip(b).map(|(&x, &y)| x | y).collect()
        }

        fn bitset_intersection_count(a: &[u64], b: &[u64]) -> u32 {
            a.iter().zip(b).map(|(&x, &y)| (x & y).count_ones()).sum()
        }

        fn bitset_remove(bs: &mut [u64], v: usize) {
            bs[v >> 6] &= !(1u64 << (v & 63));
        }

        fn bitset_add(bs: &mut [u64], v: usize) {
            bs[v >> 6] |= 1u64 << (v & 63);
        }

        fn iter_bits(bs: &[u64]) -> Vec<usize> {
            let mut out = Vec::new();
            for (wi, &w0) in bs.iter().enumerate() {
                let mut w = w0;
                while w != 0 {
                    let t = w.trailing_zeros() as usize;
                    out.push((wi << 6) + t);
                    w &= w - 1;
                }
            }
            out
        }

        fn choose_pivot(p: &[u64], x: &[u64], neighbors_bits: &[Vec<u64>]) -> Option<usize> {
            let px = bitset_or(p, x);
            if bitset_is_empty(&px) {
                return None;
            }
            let mut best_u: Option<usize> = None;
            let mut best_score: i32 = -1;

            for u in iter_bits(&px) {
                let score = bitset_intersection_count(p, &neighbors_bits[u]) as i32;
                if score > best_score {
                    best_score = score;
                    best_u = Some(u);
                }
            }
            best_u
        }

        fn make_frame(
            r: Vec<NodeId>,
            p: Vec<u64>,
            x: Vec<u64>,
            neighbors_bits: &[Vec<u64>],
        ) -> Frame {
            // Tomita pivot: pick u in P ∪ X maximizing |P ∩ N(u)|
            let candidates = if let Some(u) = choose_pivot(&p, &x, neighbors_bits) {
                // branch on vertices in P \ N(u)
                let p_without_nu = bitset_andnot(&p, &neighbors_bits[u]);
                iter_bits(&p_without_nu)
            } else {
                iter_bits(&p)
            };

            Frame {
                r,
                p,
                x,
                candidates,
                next_idx: 0,
            }
        }

        let mut out: Vec<Vec<NodeId>> = Vec::new();
        let mut stack: Vec<Frame> = vec![make_frame(r0, p0, x0, &neighbors_bits)];

        while let Some(top) = stack.last_mut() {
            // Maximal clique condition: P and X empty
            if bitset_is_empty(&top.p) && bitset_is_empty(&top.x) {
                out.push(top.r.clone());
                stack.pop();
                continue;
            }

            // Done iterating branches at this frame
            if top.next_idx >= top.candidates.len() {
                stack.pop();
                continue;
            }

            let v = top.candidates[top.next_idx];
            top.next_idx += 1;

            // If v already removed from P by earlier sibling processing, skip.
            let in_p = (top.p[v >> 6] >> (v & 63)) & 1u64 == 1;
            if !in_p {
                continue;
            }

            // Prepare child frame for recursion on v
            let mut r_child = top.r.clone();
            r_child.push(v as NodeId);

            let p_child = bitset_and(&top.p, &neighbors_bits[v]);
            let x_child = bitset_and(&top.x, &neighbors_bits[v]);

            // Move v from P to X in current frame (standard BK update)
            bitset_remove(&mut top.p, v);
            bitset_add(&mut top.x, v);

            stack.push(make_frame(r_child, p_child, x_child, &neighbors_bits));
        }

        out
    }
}
