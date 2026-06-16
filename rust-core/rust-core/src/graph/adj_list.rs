use foldhash::fast::FixedState;
use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use std::cmp::max;
use std::ops::{Deref, DerefMut, Index, IndexMut};

use duplicate::duplicate_item;
use pyo3::{FromPyObject, PyRef, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use pyo3_stub_gen::{PyStubType, impl_stub_type};
use rkyv::{Archive, Deserialize, Serialize};

use crate::graph::NodeWeight;
use crate::graph::serialize::DumpCacheToFile;

use super::types::NodeId;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[gen_stub_pyclass(module = "rust_core._core.graph")]
#[pyclass(from_py_object)]
pub struct UnweightedAdjList(pub AdjList<()>);

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[gen_stub_pyclass(module = "rust_core._core.graph")]
#[pyclass(from_py_object)]
pub struct WeightedAdjList(pub AdjList<NodeWeight>);

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
pub struct AdjList<W> {
    pub adj: Vec<Vec<(NodeId, W)>>,
    n: usize,
    m: usize,
}

impl<W> Index<usize> for AdjList<W> {
    type Output = [(NodeId, W)];

    fn index(&self, index: usize) -> &Self::Output {
        &self.adj[index]
    }
}

impl<W> IndexMut<usize> for AdjList<W> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.adj[index]
    }
}

impl<W> Index<NodeId> for AdjList<W> {
    type Output = [(NodeId, W)];

    fn index(&self, index: NodeId) -> &Self::Output {
        &self.adj[index as usize]
    }
}

impl<W> IndexMut<NodeId> for AdjList<W> {
    fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
        &mut self.adj[index as usize]
    }
}

impl<W> Default for AdjList<W> {
    fn default() -> Self {
        Self {
            adj: vec![vec![]],
            n: 0,
            m: 0,
        }
    }
}

impl<W> AdjList<W> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_nodes(n: usize) -> Self {
        Self {
            adj: (0..n).map(|_| Vec::new()).collect(),
            n,
            m: 0,
        }
    }

    pub fn from_edges_mapped(
        edges: Vec<(NodeId, NodeId, W)>,
    ) -> (AdjList<W>, Vec<NodeId>, HashMap<NodeId, NodeId, FixedState>)
    where
        W: Clone,
    {
        if edges.is_empty() {
            return (
                Self::new(),
                Vec::new(),
                HashMap::with_hasher(FixedState::default()),
            );
        }

        // Step 1: assign compact IDs
        let mut compressed_index: HashMap<NodeId, NodeId, FixedState> =
            HashMap::with_hasher(FixedState::default());
        let mut curr_number = 0;

        for (u, v, _w) in edges.iter() {
            if let Entry::Vacant(e) = compressed_index.entry(*u) {
                e.insert(curr_number);
                curr_number += 1;
            }
            if let Entry::Vacant(e) = compressed_index.entry(*v) {
                e.insert(curr_number);
                curr_number += 1;
            }
        }

        let n = curr_number as usize;

        // Step 2: compute adjacency sizes
        let mut sizes = vec![0; n];

        for (u, _, _) in edges.iter() {
            let u_idx = compressed_index[u];
            sizes[u_idx as usize] += 1;
        }

        // Step 3: allocate graph
        let mut rv = Self::with_nodes(n);

        for (u, adj) in rv.adj.iter_mut().enumerate() {
            adj.reserve(sizes[u]);
        }

        // Step 4: fill adjacency
        // (assuming add_edge handles internal cursor correctly)
        for (u, v, w) in edges.iter() {
            let u_idx = compressed_index[u];
            let v_idx = compressed_index[v];

            rv.add_edge(u_idx, v_idx, w.clone());

            // If undirected:
            // rv.add_edge(v_idx, u_idx);
        }

        // println!("Constructed adjacency list. n: {}, m: {}", rv.n(), rv.m());

        rv.m = edges.len();
        let mut original_index = Vec::with_capacity(n);
        original_index.resize(n, 0);
        for (node, &compressed) in compressed_index.iter() {
            original_index[compressed as usize] = *node;
        }

        (rv, original_index, compressed_index)
    }

    pub fn from_edges_unmapped(edges: Vec<(NodeId, NodeId, W)>) -> Self
    where
        W: Clone,
    {
        if edges.is_empty() {
            return Self::new();
        }

        let n = (edges.iter().fold(0, |acc, (u, v, w)| max(acc, max(*u, *v))) + 1) as usize;

        let mut sizes = vec![0; n];

        for (u, _, _) in edges.iter() {
            sizes[*u as usize] += 1;
        }

        let mut rv = Self::with_nodes(n);

        for (u, adj) in rv.adj.iter_mut().enumerate() {
            adj.reserve(sizes[u]);
        }

        for (u, v, w) in edges.iter() {
            rv.add_edge(*u, *v, w.clone());
        }

        rv.m = edges.len();
        rv
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn m(&self) -> usize {
        self.m
    }

    #[inline(always)]
    pub fn add_edge(&mut self, u: NodeId, v: NodeId, w: W) {
        self.adj[u as usize].push((v, w));
        self.m += 1;
    }

    #[inline(always)]
    pub fn extend(&mut self, edges: Vec<(NodeId, NodeId, W)>) {
        self.m += edges.len();
        for (u, v, w) in edges {
            self.adj[u as usize].push((v, w));
        }
    }

    pub fn remove_self_loops(&mut self) -> usize {
        let mut removed = 0;
        for (n, neighbors) in self.adj.iter_mut().enumerate() {
            neighbors.retain(|(v, _w)| {
                if *v as usize == n {
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
            neighbors.retain(|(v, _w)| {
                let v_idx = v;
                if !present.insert(*v_idx) {
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
    pub fn make_undirected(&mut self)
    where
        W: Clone,
    {
        let mut new_edges = Vec::new();

        for u in 0..self.adj.len() {
            for (v, w) in &self.adj[u] {
                new_edges.push((*v, u as NodeId, w.clone()));
            }
        }

        self.extend(new_edges);
        self.remove_multiedges();
    }

    /// Efficiently sorts each adjacency list in-place.
    /// Time Complexity: O(n + m)
    /// Space Complexity: O(n + m)
    pub fn sort_neighbors(&mut self)
    where
        W: Clone,
    {
        let n = self.n();
        let mut rv = vec![Vec::new(); n];

        for u in 0..n {
            rv[u].reserve(self[u].len());
        }

        for u in 0..n {
            for (v, w) in &self[u] {
                rv[*v as usize].push((u as NodeId, w.clone()));
            }
        }
        self.adj = rv;
    }

    pub fn iter_neighbors(&self) -> impl Iterator<Item = &Vec<(NodeId, W)>> + '_ {
        self.adj.iter()
    }
}

#[duplicate_item(
    adj_type            weight_type  tuple_edge_type;
    [UnweightedAdjList] [()]         [(NodeId, NodeId)];
    [WeightedAdjList]   [NodeWeight] [(NodeId, NodeId, NodeWeight)];
)]
#[gen_stub_pymethods(module = "rust_core._core.graph")]
#[pymethods]
impl adj_type {
    #[new]
    pub fn new() -> Self {
        Self(AdjList::new())
    }

    #[staticmethod]
    pub fn with_nodes(n: usize) -> Self {
        Self(AdjList::with_nodes(n))
    }

    #[staticmethod]
    pub fn from_edges_mapped(
        edges: Vec<tuple_edge_type>,
    ) -> (adj_type, Vec<NodeId>, HashMap<NodeId, NodeId, FixedState>) {
        let edges = Self::normalize_edge_list(edges);

        let (adj, m1, m2) = AdjList::<weight_type>::from_edges_mapped(edges);
        (Self(adj), m1, m2)
    }

    #[staticmethod]
    pub fn from_edges_unmapped(edges: Vec<tuple_edge_type>) -> Self {
        let edges = Self::normalize_edge_list(edges);
        Self(AdjList::from_edges_unmapped(edges))
    }

    pub fn remove_self_loops(&mut self) -> usize {
        self.0.remove_self_loops()
    }

    pub fn remove_multiedges(&mut self) -> usize {
        self.0.remove_multiedges()
    }

    // The presence of multiesges makes this operation ambiguous. Multiedges are hence removed
    pub fn make_undirected(&mut self) {
        self.0.make_undirected()
    }

    /// Efficiently sorts each adjacency list in-place.
    /// Time Complexity: O(n + m)
    /// Space Complexity: O(n + m)
    pub fn sort_neighbors(&mut self) {
        self.0.sort_neighbors()
    }
}

impl UnweightedAdjList {
    pub(crate) fn normalize_edge_list(edges: Vec<(NodeId, NodeId)>) -> Vec<(NodeId, NodeId, ())> {
        edges.into_iter().map(|(u, v)| (u, v, ())).collect()
    }
}

impl WeightedAdjList {
    pub(crate) fn normalize_edge_list(
        edges: Vec<(NodeId, NodeId, NodeWeight)>,
    ) -> Vec<(NodeId, NodeId, NodeWeight)> {
        edges
    }
}

#[derive(FromPyObject)]
pub enum PyAdjList<'py> {
    Weighted(PyRef<'py, WeightedAdjList>),
    Unweighted(PyRef<'py, UnweightedAdjList>),
}
impl_stub_type!(PyAdjList<'_> = WeightedAdjList | UnweightedAdjList);

impl Deref for UnweightedAdjList {
    type Target = AdjList<()>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for UnweightedAdjList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for WeightedAdjList {
    type Target = AdjList<NodeWeight>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for WeightedAdjList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// struct UnweightedAdjListIter<'a> {
//     adj_list: &'a UnweightedAdjList,
//     node_idx: usize,
// }
//
// struct WeightedAdjListIter<'a> {
//     adj_list: &'a WeightedAdjList,
//     node_idx: usize,
// }
//
// impl<'a> Iterator for UnweightedAdjListIter<'a> {
//     type Item = std::slice::Iter<'a, (NodeId)>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.node_idx >= self.adj_list.n() {
//             return None;
//         }
//
//         let current_neighbors = &self.adj_list.adj[self.node_idx];
//         self.node_idx += 1;
//
//         let neighbors: Vec<NodeId> = current_neighbors.iter().map(|(v, _w)| *v).collect();
//
//         // Wrap the iterator in a Box
//         Some(neighbors.iter())
//     }
// }
//
// impl<'a> Iterator for WeightedAdjListIter<'a> {
//     type Item = std::slice::Iter<'a, (NodeId, NodeWeight)>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.node_idx >= self.adj_list.n() {
//             return None;
//         }
//
//         let current_neighbors = &self.adj_list.adj[self.node_idx];
//         self.node_idx += 1;
//
//         Some(current_neighbors.iter())
//     }
// }
