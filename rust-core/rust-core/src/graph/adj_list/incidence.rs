use foldhash::fast::FixedState;
use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use rust_core_macros::hoist_mod;
use std::cmp::max;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut, Range, RangeBounds};

use duplicate::duplicate_item;
use rkyv::{Archive, Deserialize, Serialize};

use crate::graph::base::incidence::{ListNeighbor, SetNeighbor};
use crate::graph::base::{AdjBase, Directed, Directionality, Neighbor, Undirected};
use crate::graph::serialize::DumpCacheToFile;
use crate::graph::{EdgeId, NodeWeight};

use crate::graph::hyperedge::NodeId;

/// A incidence list where each node's neighbors is stored in a Vec.
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
pub struct IncList<W, D = Undirected>
where
    D: Directionality,
{
    adj: Vec<Vec<ListNeighbor<NodeId, EdgeId, W>>>,
    n: usize,
    m: usize,

    next_edge_id: EdgeId,

    _directionality_marker: PhantomData<D>,
}

impl<W, D> IncList<W, D>
where
    D: Directionality,
    Self: AdjBase<W>,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_nodes(n: usize) -> Self {
        Self {
            adj: (0..n).map(|_| Vec::new()).collect(),
            n,
            m: 0,
            next_edge_id: 0,
            _directionality_marker: PhantomData,
        }
    }

    pub fn from_edges_mapped(
        edges: Vec<(NodeId, NodeId, W)>,
    ) -> (
        IncList<W, D>,
        Vec<NodeId>,
        HashMap<NodeId, NodeId, FixedState>,
    )
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

        for (u, v, _) in edges.iter() {
            let u_idx = compressed_index[u];
            let v_idx = compressed_index[v];
            sizes[u_idx as usize] += 1;
            sizes[v_idx as usize] += 1;
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

    pub fn remove_self_loops(&mut self) -> usize {
        let mut removed = 0;
        for (x, neighbors) in self.adj.iter_mut().enumerate() {
            neighbors.retain(|neigbor| {
                if neigbor.node as usize == x {
                    removed += 1;
                    false
                } else {
                    true
                }
            });
        }
        self.m -= removed;
        removed
    }

    pub fn remove_multiedges(&mut self) -> usize {
        let mut removed = 0;
        let mut present = HashSet::new();

        for neighbors in self.adj.iter_mut() {
            present.clear();
            neighbors.retain(|neighbor| {
                if !present.insert(neighbor.node) {
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

    /// Efficiently sorts each adjacency list.
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
            for neighbor in self.adj[u].drain(..) {
                rv[neighbor.node as usize].push(ListNeighbor::new(
                    u as NodeId,
                    neighbor.weight,
                    neighbor.edge,
                ));
            }
        }
        self.adj = rv;
    }

    pub fn iter_neighbors(
        &self,
    ) -> impl Iterator<Item = &Vec<ListNeighbor<NodeId, EdgeId, W>>> + '_ {
        self.adj.iter()
    }

    pub fn iter_neighbors_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut Vec<ListNeighbor<NodeId, EdgeId, W>>> + '_ {
        self.adj.iter_mut()
    }

    pub fn drain_neighbors(
        &mut self,
        range: impl RangeBounds<usize>,
    ) -> impl Iterator<Item = Vec<ListNeighbor<NodeId, EdgeId, W>>> + '_ {
        self.adj.drain(range)
    }

    pub fn compact_edge_ids(&mut self) {
        let mut ids = vec![HashMap::with_hasher(FixedState::default()); self.n()];
        self.next_edge_id = 0;
        for (x, neighbors) in self.adj.iter_mut().enumerate() {
            for neighbor in neighbors.iter_mut() {
                match ids[x].get(&neighbor.edge) {
                    Some(edge_id) => neighbor.edge = *edge_id,
                    None => {
                        neighbor.edge = self.next_edge_id;
                        ids[neighbor.node as usize].insert(x as NodeId, neighbor.edge);
                        self.next_edge_id += 1;
                    }
                }

                // neighbor.edge = self.next_edge_id;
                // self.next_edge_id += 1;
            }
        }
    }
}

impl<W> AdjBase<W> for IncList<W, Undirected> {
    #[inline(always)]
    fn add_edge(&mut self, u: NodeId, v: NodeId, w: W)
    where
        W: Clone,
    {
        self.adj[u as usize].push(ListNeighbor::new(v, w.clone(), self.next_edge_id));
        self.adj[v as usize].push(ListNeighbor::new(u, w, self.next_edge_id));
        self.next_edge_id += 1;
        self.m += 1;
    }

    #[inline(always)]
    fn remove_edges_between(&mut self, u: NodeId, v: NodeId) {
        self.adj[u as usize].retain(|e| e.node != v);
        self.adj[v as usize].retain(|e| e.node != u);
    }
}

impl<W> IncList<W, Undirected> {
    // The presence of multiesges makes this operation ambiguous. Multiedges are hence removed
    pub fn into_directed(mut self) -> IncList<W, Directed> {
        let mut next_edge_id = 0;
        for neighbors in self.iter_neighbors_mut() {
            for neighbor in neighbors.iter_mut() {
                neighbor.edge = next_edge_id;
                next_edge_id += 1;
            }
        }

        IncList {
            adj: self.adj,
            n: self.n,
            m: self.m,
            next_edge_id: next_edge_id,
            _directionality_marker: PhantomData,
        }
    }
}

impl<W> IncList<W, Directed> {
    /// Makes the directed graph undirected by adding an edge (v,u) for each edge (u,v). If
    /// `allow_multiedges` is set to `false`, it will remove any multiedges that may arise from this
    /// operation.
    pub fn into_undirected(mut self, allow_multiedges: bool) -> IncList<W, Undirected>
    where
        W: Clone,
    {
        let mut rv = IncList::<W, Undirected>::with_nodes(self.n());
        let mut next_edge_id = 0;
        for (x, neighbors) in self.drain_neighbors(..).enumerate() {
            rv.adj[x].reserve(neighbors.len()); // Does not reserve all required space
            // bu it should be enough to avoid too many reallocations

            for neighbor in neighbors.into_iter() {
                rv.add_edge(x as NodeId, neighbor.node, neighbor.weight);
            }
        }

        if !allow_multiedges {
            rv.remove_multiedges();
        }

        rv
    }
}

impl<W> AdjBase<W> for IncList<W, Directed> {
    #[inline(always)]
    fn add_edge(&mut self, u: NodeId, v: NodeId, w: W) {
        self.adj[u as usize].push(ListNeighbor::new(v, w, self.next_edge_id));
        self.next_edge_id += 1;
        self.m += 1;
    }

    #[inline(always)]
    fn remove_edges_between(&mut self, u: NodeId, v: NodeId) {
        self.adj[u as usize].retain(|e| e.node != v);
    }
}

impl<W, D> Index<usize> for IncList<W, D>
where
    D: Directionality,
{
    type Output = Vec<ListNeighbor<NodeId, EdgeId, W>>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.adj[index]
    }
}

impl<W, D> IndexMut<usize> for IncList<W, D>
where
    D: Directionality,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.adj[index]
    }
}

impl<W, D> Index<NodeId> for IncList<W, D>
where
    D: Directionality,
{
    type Output = Vec<ListNeighbor<NodeId, EdgeId, W>>;

    fn index(&self, index: NodeId) -> &Self::Output {
        &self.adj[index as usize]
    }
}

impl<W, D> IndexMut<NodeId> for IncList<W, D>
where
    D: Directionality,
{
    fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
        &mut self.adj[index as usize]
    }
}

impl<W, D> Default for IncList<W, D>
where
    D: Directionality,
{
    fn default() -> Self {
        Self {
            adj: Vec::new(),
            n: 0,
            m: 0,
            next_edge_id: 0,
            _directionality_marker: PhantomData,
        }
    }
}

/// A incidence list where each node's neighbors is stored in a HashMap.
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
pub struct IncSet<W, D = Undirected>
where
    D: Directionality,
{
    adj: Vec<HashMap<NodeId, SetNeighbor<EdgeId, W>, FixedState>>,
    n: usize,
    m: usize,

    next_edge_id: EdgeId,

    _directionality_marker: PhantomData<D>,
}

impl<W, D> IncSet<W, D>
where
    D: Directionality,
    Self: AdjBase<W>,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_nodes(n: usize) -> Self {
        Self {
            adj: (0..n)
                .map(|_| HashMap::with_hasher(FixedState::default()))
                .collect(),
            n,
            m: 0,
            next_edge_id: 0,
            _directionality_marker: PhantomData,
        }
    }

    pub fn from_edges_mapped(
        edges: Vec<(NodeId, NodeId, W)>,
    ) -> (
        IncSet<W, D>,
        Vec<NodeId>,
        HashMap<NodeId, NodeId, FixedState>,
    )
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

        let mut rv = Self::with_nodes(n);

        for (u, v, w) in edges.iter() {
            let u_idx = compressed_index[u];
            let v_idx = compressed_index[v];
            rv.add_edge(u_idx, v_idx, w.clone());
        }

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

        let n = (edges
            .iter()
            .fold(0, |acc, (u, v, _w)| max(acc, max(*u, *v)))
            + 1) as usize;

        let mut rv = Self::with_nodes(n);

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

    pub fn remove_self_loops(&mut self) -> usize {
        let mut removed = 0;
        for (x, neighbors) in self.adj.iter_mut().enumerate() {
            if neighbors.remove(&(x as NodeId)).is_some() {
                removed += 1;
            }
        }
        self.m -= removed;
        removed
    }

    pub fn remove_multiedges(&mut self) -> usize {
        // The underlying HashMap already prevents duplicate edges.
        0
    }

    pub fn iter_neighbors(
        &self,
    ) -> impl Iterator<Item = &HashMap<NodeId, SetNeighbor<EdgeId, W>, FixedState>> + '_ {
        self.adj.iter()
    }

    pub fn iter_neighbors_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut HashMap<NodeId, SetNeighbor<EdgeId, W>, FixedState>> + '_ {
        self.adj.iter_mut()
    }

    pub fn drain_neighbors(
        &mut self,
        range: impl RangeBounds<usize>,
    ) -> impl Iterator<Item = HashMap<NodeId, SetNeighbor<EdgeId, W>, FixedState>> + '_ {
        self.adj.drain(range)
    }

    pub fn compact_edge_ids(&mut self) {
        self.next_edge_id = 0;
        for neighbors in self.adj.iter_mut() {
            for (_v, neighbor) in neighbors.iter_mut() {
                neighbor.edge = self.next_edge_id;
                self.next_edge_id += 1;
            }
        }
    }
}

impl<W> AdjBase<W> for IncSet<W, Undirected> {
    #[inline(always)]
    fn add_edge(&mut self, u: NodeId, v: NodeId, w: W)
    where
        W: Clone,
    {
        self.adj[u as usize].insert(v, SetNeighbor::new(w.clone(), self.next_edge_id));
        self.adj[v as usize].insert(u, SetNeighbor::new(w, self.next_edge_id));
        self.next_edge_id += 1;
        self.m += 1;
    }

    #[inline(always)]
    fn remove_edges_between(&mut self, u: NodeId, v: NodeId) {
        self.adj[u as usize].remove(&v);
        self.adj[v as usize].remove(&u);
    }
}

impl<W> IncSet<W, Undirected> {
    /// Convert from undirected to directed by reassigning all edge IDs.
    pub fn into_directed(mut self) -> IncSet<W, Directed> {
        for neighbors in self.adj.iter_mut() {
            for (_v, neighbor) in neighbors.iter_mut() {
                neighbor.edge = self.next_edge_id;
                self.next_edge_id += 1;
            }
        }
        IncSet {
            adj: self.adj,
            n: self.n,
            m: self.m,
            next_edge_id: self.next_edge_id,
            _directionality_marker: PhantomData,
        }
    }
}

impl<W> AdjBase<W> for IncSet<W, Directed> {
    #[inline(always)]
    fn add_edge(&mut self, u: NodeId, v: NodeId, w: W)
    where
        W: Clone,
    {
        self.adj[u as usize].insert(v, SetNeighbor::new(w, self.next_edge_id));
        self.next_edge_id += 1;
        self.m += 1;
    }

    #[inline(always)]
    fn remove_edges_between(&mut self, u: NodeId, v: NodeId) {
        self.adj[u as usize].remove(&v);
    }
}

impl<W> IncSet<W, Directed> {
    /// Convert from directed to undirected by adding the reverse edges.
    pub fn into_undirected(mut self, allow_multiedges: bool) -> IncSet<W, Undirected>
    where
        W: Clone,
    {
        let mut rv = IncSet::<W, Undirected>::with_nodes(self.n());
        for (x, neighbors) in self.drain_neighbors(..).enumerate() {
            for (v, neighbor) in neighbors.into_iter() {
                rv.add_edge(x as NodeId, v, neighbor.weight);
            }
        }
        if !allow_multiedges {
            rv.remove_multiedges();
        }
        rv
    }
}

impl<W, D> Index<usize> for IncSet<W, D>
where
    D: Directionality,
{
    type Output = HashMap<NodeId, SetNeighbor<EdgeId, W>, FixedState>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.adj[index]
    }
}

impl<W, D> IndexMut<usize> for IncSet<W, D>
where
    D: Directionality,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.adj[index]
    }
}

impl<W, D> Index<NodeId> for IncSet<W, D>
where
    D: Directionality,
{
    type Output = HashMap<NodeId, SetNeighbor<EdgeId, W>, FixedState>;

    fn index(&self, index: NodeId) -> &Self::Output {
        &self.adj[index as usize]
    }
}

impl<W, D> IndexMut<NodeId> for IncSet<W, D>
where
    D: Directionality,
{
    fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
        &mut self.adj[index as usize]
    }
}

impl<W, D> Default for IncSet<W, D>
where
    D: Directionality,
{
    fn default() -> Self {
        Self {
            adj: Vec::new(),
            n: 0,
            m: 0,
            next_edge_id: 0,
            _directionality_marker: PhantomData,
        }
    }
}

/*
#[cfg(feature = "bindings")]
#[hoist_mod]
mod bindings {
    use pyo3::{FromPyObject, PyRef, pyclass, pymethods};
    use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
    use pyo3_stub_gen::{PyStubType, impl_stub_type};

    #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
    #[gen_stub_pyclass(module = "rust_core._core.graph")]
    #[pyclass(from_py_object)]
    pub struct UnweightedAdjList(pub IncList<()>);

    #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
    #[gen_stub_pyclass(module = "rust_core._core.graph")]
    #[pyclass(from_py_object)]
    pub struct WeightedAdjList(pub IncList<NodeWeight>);

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
            Self(IncList::new())
        }

        #[staticmethod]
        pub fn with_nodes(n: usize) -> Self {
            Self(IncList::with_nodes(n))
        }

        #[staticmethod]
        pub fn from_edges_mapped(
            edges: Vec<tuple_edge_type>,
        ) -> (adj_type, Vec<NodeId>, HashMap<NodeId, NodeId, FixedState>) {
            let edges = Self::normalize_edge_list(edges);

            let (adj, m1, m2) = IncList::<weight_type>::from_edges_mapped(edges);
            (Self(adj), m1, m2)
        }

        #[staticmethod]
        pub fn from_edges_unmapped(edges: Vec<tuple_edge_type>) -> Self {
            let edges = Self::normalize_edge_list(edges);
            Self(IncList::from_edges_unmapped(edges))
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
        pub(crate) fn normalize_edge_list(
            edges: Vec<(NodeId, NodeId)>,
        ) -> Vec<(NodeId, NodeId, ())> {
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
        type Target = IncList<()>;

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
        type Target = IncList<NodeWeight>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl DerefMut for WeightedAdjList {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    // ------------------------------------------------------------------------
    // AdjSet bindings (same functionality, HashMap‑backed)
    // ------------------------------------------------------------------------

    #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
    #[gen_stub_pyclass(module = "rust_core._core.graph")]
    #[pyclass(from_py_object)]
    pub struct UnweightedAdjSet(pub IncSet<()>);

    #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
    #[gen_stub_pyclass(module = "rust_core._core.graph")]
    #[pyclass(from_py_object)]
    pub struct WeightedAdjSet(pub IncSet<NodeWeight>);

    #[duplicate_item(
    adj_type            weight_type  tuple_edge_type;
    [UnweightedAdjSet] [()]         [(NodeId, NodeId)];
    [WeightedAdjSet]   [NodeWeight] [(NodeId, NodeId, NodeWeight)];
)]
    #[gen_stub_pymethods(module = "rust_core._core.graph")]
    #[pymethods]
    impl adj_type {
        #[new]
        pub fn new() -> Self {
            Self(IncSet::new())
        }

        #[staticmethod]
        pub fn with_nodes(n: usize) -> Self {
            Self(IncSet::with_nodes(n))
        }

        #[staticmethod]
        pub fn from_edges_mapped(
            edges: Vec<tuple_edge_type>,
        ) -> (adj_type, Vec<NodeId>, HashMap<NodeId, NodeId, FixedState>) {
            let edges = Self::normalize_edge_list(edges);
            let (adj, m1, m2) = IncSet::<weight_type>::from_edges_mapped(edges);
            (Self(adj), m1, m2)
        }

        #[staticmethod]
        pub fn from_edges_unmapped(edges: Vec<tuple_edge_type>) -> Self {
            let edges = Self::normalize_edge_list(edges);
            Self(IncSet::from_edges_unmapped(edges))
        }

        pub fn remove_self_loops(&mut self) -> usize {
            self.0.remove_self_loops()
        }

        pub fn remove_multiedges(&mut self) -> usize {
            self.0.remove_multiedges()
        }

        /// No‑op for IncSet because the underlying HashMap already makes
        /// multi‑edges impossible, and ordering is irrelevant.
        pub fn sort_neighbors(&mut self) {
            self.0.sort_neighbors()
        }
    }

    impl UnweightedAdjSet {
        pub(crate) fn normalize_edge_list(
            edges: Vec<(NodeId, NodeId)>,
        ) -> Vec<(NodeId, NodeId, ())> {
            edges.into_iter().map(|(u, v)| (u, v, ())).collect()
        }
    }

    impl WeightedAdjSet {
        pub(crate) fn normalize_edge_list(
            edges: Vec<(NodeId, NodeId, NodeWeight)>,
        ) -> Vec<(NodeId, NodeId, NodeWeight)> {
            edges
        }
    }

    #[derive(FromPyObject)]
    pub enum PyAdjSet<'py> {
        Weighted(PyRef<'py, WeightedAdjSet>),
        Unweighted(PyRef<'py, UnweightedAdjSet>),
    }

    impl_stub_type!(PyAdjSet<'_> = WeightedAdjSet | UnweightedAdjSet);

    impl Deref for UnweightedAdjSet {
        type Target = IncSet<()>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl DerefMut for UnweightedAdjSet {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl Deref for WeightedAdjSet {
        type Target = IncSet<NodeWeight>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl DerefMut for WeightedAdjSet {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }
}
*/
