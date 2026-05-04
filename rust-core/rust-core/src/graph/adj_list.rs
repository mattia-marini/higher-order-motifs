use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::ops::{Index, IndexMut};

use pyo3::{pyclass, pymethods};
use pyo3_stub_gen::PyStubType;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rkyv::{Archive, Deserialize, Serialize};

use super::types::NodeId;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
#[pyclass]
pub struct AdjList {
    pub adj: Vec<Vec<NodeId>>,
    n: usize,
    m: usize,
}

impl PyStubType for &mut AdjList {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        AdjList::type_output()
    }
}

impl Index<usize> for AdjList {
    type Output = [NodeId];

    fn index(&self, index: usize) -> &Self::Output {
        &self.adj[index]
    }
}

impl IndexMut<usize> for AdjList {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.adj[index]
    }
}

impl Index<NodeId> for AdjList {
    type Output = [NodeId];

    fn index(&self, index: NodeId) -> &Self::Output {
        &self.adj[index as usize]
    }
}

impl IndexMut<NodeId> for AdjList {
    fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
        &mut self.adj[index as usize]
    }
}

impl Default for AdjList {
    fn default() -> Self {
        Self::new()
    }
}

#[gen_stub_pymethods(module = "rust_core.core.graph")]
#[pymethods]
impl AdjList {
    #[new]
    pub fn new() -> Self {
        Self {
            adj: Vec::new(),
            m: 0,
            n: 0,
        }
    }

    #[staticmethod]
    pub fn with_nodes(n: usize) -> Self {
        let mut adj = Vec::new();
        adj.resize(n, Vec::new());
        Self { adj, n, m: 0 }
    }

    // #[staticmethod]
    // pub fn from_mat(adj_mat: AdjMat) -> Self {
    //     let mut adj_list = Self::with_nodes(adj_mat.n());
    //     for i in 0..adj_mat.n() {
    //         for j in adj_mat.mat[i].iter() {
    //             adj_list.adj[j].push(i as NodeId);
    //         }
    //     }
    //     adj_list
    // }

    #[staticmethod]
    pub fn from_edges(edges: Vec<(NodeId, NodeId)>) -> Self {
        if edges.is_empty() {
            return Self::new();
        }

        // Step 1: assign compact IDs
        let mut node_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut curr_number = 0;

        for (u, v) in edges.iter() {
            if let std::collections::hash_map::Entry::Vacant(e) = node_map.entry(*u) {
                e.insert(curr_number);
                curr_number += 1;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = node_map.entry(*v) {
                e.insert(curr_number);
                curr_number += 1;
            }
        }

        let n = curr_number as usize;

        // Step 2: compute adjacency sizes
        let mut sizes = vec![0; n];

        for (u, _) in edges.iter() {
            let u_idx = node_map[&u];
            sizes[u_idx as usize] += 1;
        }

        // Step 3: allocate graph
        let mut rv = Self::with_nodes(n);

        for (u, adj) in rv.adj.iter_mut().enumerate() {
            adj.reserve(sizes[u]);
        }

        // Step 4: fill adjacency
        // (assuming add_edge handles internal cursor correctly)
        for (u, v) in edges.iter() {
            let u_idx = node_map[u];
            let v_idx = node_map[v];

            rv.add_edge(u_idx, v_idx);

            // If undirected:
            // rv.add_edge(v_idx, u_idx);
        }

        // println!("Constructed adjacency list. n: {}, m: {}", rv.n(), rv.m());

        rv.m = edges.len();
        rv
    }

    #[staticmethod]
    pub fn from_edges_unmapped(edges: Vec<(NodeId, NodeId)>) -> Self {
        if edges.is_empty() {
            return Self::new();
        }

        let n = (edges.iter().fold(0, |acc, (u, v)| max(acc, max(*u, *v))) + 1) as usize;

        let mut sizes = vec![0; n];

        for (u, _) in edges.iter() {
            sizes[*u as usize] += 1;
        }

        let mut rv = Self::with_nodes(n);

        for (u, adj) in rv.adj.iter_mut().enumerate() {
            adj.reserve(sizes[u]);
        }

        for (u, v) in edges.iter() {
            rv.add_edge(*u, *v);
        }

        rv.m = edges.len();
        rv
    }

    pub fn remove_self_loops(&mut self) -> usize {
        let mut removed = 0;
        for (n, neighbors) in self.adj.iter_mut().enumerate() {
            neighbors.retain(|&v| {
                if v as usize == n {
                    removed += 1;
                    false
                } else {
                    true
                }
            });
        }
        // println!("Removed {} self-loops", removed);
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
        // println!("Removed {} multiedges", removed);
        self.m -= removed;
        removed
    }

    // The presence of multiesges makes this operation ambiguous. Multiedges are hence removed
    pub fn make_undirected(&mut self) {
        let mut new_edges = Vec::new();

        for u in 0..self.adj.len() {
            for &v in &self.adj[u] {
                new_edges.push((v, u as NodeId));
            }
        }

        // println!("TESET");
        self.extend(new_edges);
        self.remove_multiedges();
    }

    /// Efficiently sorts each adjacency list in-place.
    /// Time Complexity: O(n + m)
    /// Space Complexity: O(n + m)
    pub fn sort_neighbors(&mut self) {
        let n = self.n();
        let mut rv = vec![Vec::new(); n];

        for u in 0..n {
            rv[u].reserve(self[u].len());
        }

        for u in 0..n {
            for &v in &self[u] {
                rv[v as usize].push(u as NodeId);
            }
        }
        self.adj = rv;
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn m(&self) -> usize {
        self.m
    }

    #[inline(always)]
    pub fn add_edge(&mut self, u: NodeId, v: NodeId) {
        self.adj[u as usize].push(v);
        self.m += 1;
    }

    #[inline(always)]
    pub fn extend(&mut self, edges: Vec<(NodeId, NodeId)>) {
        self.m += edges.len();
        for (u, v) in edges {
            self.adj[u as usize].push(v);
        }
    }
}
