use bit_set::BitSet;
use num_traits::{AsPrimitive, One, PrimInt, Unsigned, Zero};

use rkyv::{
    Archive, Deserialize, Serialize,
    bytecheck::CheckBytes,
    de::Pool,
    deserialize,
    rancor::Strategy,
    util::AlignedVec,
    validation::{Validator, archive::ArchiveValidator, shared::SharedValidator},
};

use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    fs::File,
    hash::Hash,
    io::Write,
    ops::{AddAssign, Index},
    path::Path,
};

use super::adj_mat::AdjMat;

pub struct AdjList<E> {
    pub adj: Vec<Vec<E>>,
    n: usize,
    m: usize,
}

impl<E> AdjList<E>
where
    E: Zero + Clone + Copy + AsPrimitive<usize> + Ord + Hash + Eq + One + AddAssign,
    usize: AsPrimitive<E>,
{
    pub fn new() -> Self {
        Self {
            adj: Vec::new(),
            m: 0,
            n: 0,
        }
    }

    pub fn with_nodes(n: usize) -> Self {
        let mut adj = Vec::new();
        adj.resize(n, Vec::new());
        Self { adj, n, m: 0 }
    }

    pub fn from_mat(adj_mat: &AdjMat) -> Self {
        let mut adj_list = Self::with_nodes(adj_mat.n());
        for i in 0..adj_mat.n() {
            for j in adj_mat.mat[i].iter() {
                adj_list.adj[j].push(i.as_());
            }
        }
        adj_list
    }

    pub fn from_edges(edges: &[(E, E)]) -> Self {
        if edges.is_empty() {
            return Self::new();
        }

        // Step 1: assign compact IDs
        let mut node_map: HashMap<E, E> = HashMap::new();
        let mut curr_number = E::zero();

        for &(u, v) in edges {
            if !node_map.contains_key(&u) {
                node_map.insert(u, curr_number);
                curr_number += E::one();
            }
            if !node_map.contains_key(&v) {
                node_map.insert(v, curr_number);
                curr_number += E::one();
            }
        }

        let n = curr_number.as_();

        // Step 2: compute adjacency sizes
        let mut sizes = vec![0usize; n];

        for &(u, _) in edges {
            let u_idx = node_map[&u].as_();
            sizes[u_idx] += 1;
        }

        // Step 3: allocate graph
        let mut rv = Self::with_nodes(n);

        for (u, adj) in rv.adj.iter_mut().enumerate() {
            adj.reserve(sizes[u]);
        }

        // Step 4: fill adjacency
        // (assuming add_edge handles internal cursor correctly)
        for &(u, v) in edges {
            let u_idx = node_map[&u];
            let v_idx = node_map[&v];

            rv.add_edge(u_idx, v_idx);

            // If undirected:
            // rv.add_edge(v_idx, u_idx);
        }

        println!("Constructed adjacency list. n: {}, m: {}", rv.n(), rv.m());

        rv.m = edges.len();
        rv
    }

    pub fn remove_self_loops(&mut self) -> usize {
        let mut removed = 0;
        for (n, neighbors) in self.adj.iter_mut().enumerate() {
            neighbors.retain(|&v| {
                if v.as_() == n {
                    removed += 1;
                    false
                } else {
                    true
                }
            });
        }
        println!("Removed {} self-loops", removed);
        self.m -= removed;
        removed
    }

    pub fn remove_multiedges(&mut self) -> usize {
        let mut removed = 0;
        let mut present = HashSet::new();

        for neighbors in self.adj.iter_mut() {
            present.clear();
            neighbors.retain(|&v| {
                let v_idx = v;
                if !present.insert(v_idx) {
                    removed += 1;
                    false
                } else {
                    true
                }
            });
        }
        println!("Removed {} multiedges", removed);
        self.m -= removed;
        removed
    }

    // The presence of multiesges makes this operation ambiguous. Multiedges are hence removed
    pub fn make_undirected(&mut self) {
        let mut new_edges = Vec::new();

        for u in 0..self.adj.len() {
            for &v in &self.adj[u] {
                new_edges.push((v, u.as_()));
            }
        }

        self.extend(new_edges);
        self.remove_multiedges();
        // self.m = self.adj.iter().map(|neighbors| neighbors.len()).sum();
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn m(&self) -> usize {
        self.m
    }

    #[inline(always)]
    pub fn add_edge(&mut self, u: E, v: E) {
        self.adj[u.as_()].push(v);
        self.m += 1;
    }

    #[inline(always)]
    pub fn extend(&mut self, edges: Vec<(E, E)>) {
        self.m += edges.len();
        for (u, v) in edges {
            self.adj[u.as_()].push(v);
        }
    }
}
