use ouroboros::self_referencing;
use pyo3::PyErr;
use pyo3::exceptions::PyValueError;
use seq_macro::seq;
use std::hash::Hash;
use std::{collections::HashMap, fs::File, io::Write, path::Path};

use duplicate::duplicate_item;
use rust_core_macros::{ct_map_accessor, inherent, repeat};

use pyo3::{
    Bound, FromPyObject, IntoPyObject, PyRef, PyResult, Python, pyclass, pymethods,
    types::{PyAnyMethods, PySet, PyTuple},
};
use pyo3_stub_gen::{
    PyStubType,
    derive::{gen_stub_pyclass, gen_stub_pymethods},
    impl_stub_type, type_alias,
};

use rkyv::{
    Archive, Deserialize, Serialize, collections::swiss_table::ArchivedHashSet, rend::u32_le,
};

use rust_core_macros::ct_map;

use crate::graph::edge_collection::{HashSet, MAX_HX_SIZE, MIN_HX_SIZE};
use crate::graph::error::GraphError;
use crate::graph::types2::{NodeId, NodeWeight};
use crate::graph::{edge_collection::StaticEdgeSet, types2::Hx};

// #[derive(Archive, Serialize, Deserialize, Debug, PartialEq, Eq)]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// #[pyclass]
// #[pyo3(get)]
pub struct Hypergraph<T, W> {
    pub edge_set: StaticEdgeSet<T, W>,
    pub nodes: HashMap<T, usize>,

    n: usize,
    m: usize,
}

pub trait HypergraphAccessor<const N: usize, T, W> {
    fn edges(&self) -> &HashSet<Hx<N, T, W>>;
    fn edges_mut(&mut self) -> &mut HashSet<Hx<N, T, W>>;
    fn take_edges(&mut self) -> HashSet<Hx<N, T, W>>;
}

#[repeat(rg(2..11))]
#[repeat_item]
impl<T, W> HypergraphAccessor<N, T, W> for Hypergraph<T, W> {
    fn edges(&self) -> &HashSet<Hx<N, T, W>> {
        self.edge_set.get_bucket::<N>()
    }

    fn edges_mut(&mut self) -> &mut HashSet<Hx<N, T, W>> {
        self.edge_set.get_bucket_mut::<N>()
    }

    fn take_edges(&mut self) -> HashSet<Hx<N, T, W>> {
        self.edge_set.take_bucket::<N>()
    }
}

// #[pymethods]
impl<T, W> Hypergraph<T, W> {
    // #[new]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            n: 0,
            m: 0,
            edge_set: StaticEdgeSet::new(),
        }
    }

    pub fn edges<const N: usize>(&self) -> &HashSet<Hx<N, T, W>>
    where
        Self: HypergraphAccessor<N, T, W>,
    {
        <Self as HypergraphAccessor<N, T, W>>::edges(self)
    }

    pub fn edges_mut<const N: usize>(&mut self) -> &mut HashSet<Hx<N, T, W>>
    where
        Self: HypergraphAccessor<N, T, W>,
    {
        <Self as HypergraphAccessor<N, T, W>>::edges_mut(self)
    }

    pub fn take_edges<const N: usize>(&mut self) -> HashSet<Hx<N, T, W>>
    where
        Self: HypergraphAccessor<N, T, W>,
    {
        <Self as HypergraphAccessor<N, T, W>>::take_edges(self)
    }

    fn update_n<const N: usize>(&mut self, edge: &[T; N], add: bool)
    where
        T: Hash + Eq + Clone,
        Self: HypergraphAccessor<N, T, W>,
    {
        let inc = if add { 1 } else { -1 };

        for n in edge {
            self.nodes
                .entry(n.clone())
                .and_modify(|degree| {
                    *degree = (*degree as i64 + inc) as usize;
                })
                .or_insert_with(|| {
                    self.n = (self.n as i64 + 1) as usize;
                    1
                });
        }
    }

    pub fn add_edge<const N: usize>(&mut self, edge: Hx<N, T, W>) -> bool
    where
        T: Hash + Eq + Clone,
        Self: HypergraphAccessor<N, T, W>,
    {
        if !self.edges::<N>().contains(&edge) {
            self.update_n(&edge.nodes, true);
            self.m += 1;
            self.edges_mut::<N>().insert(edge)
        } else {
            false
        }
    }

    pub fn remove_edge<const N: usize>(&mut self, edge: &[T; N]) -> bool
    where
        T: Hash + Eq + Clone + Ord,
        Self: HypergraphAccessor<N, T, W>,
    {
        let mut edge = edge.clone();
        edge.sort_unstable();
        self.remove_edge_unchecked(&edge)
    }

    pub fn remove_edge_unchecked<const N: usize>(&mut self, edge: &[T; N]) -> bool
    where
        T: Hash + Eq + Clone,
        Self: HypergraphAccessor<N, T, W>,
    {
        if self.edges::<N>().contains(edge) {
            self.update_n(&edge, false);
            self.m -= 1;
            self.edges_mut::<N>().remove(edge)
        } else {
            false
        }
    }

    // #[staticmethod]
    /// Returns number of edges successfully added
    pub fn extend_with_edges<const N: usize>(&mut self, edges: Vec<Hx<N, T, W>>) -> usize
    where
        T: Hash + Eq + Clone,
        Self: HypergraphAccessor<N, T, W>,
    {
        let edge_set = self.edges_mut::<N>();
        let mut added = 0;
        for edge in edges.into_iter() {
            added += self.add_edge(edge) as usize;
        }
        added
    }

    pub fn has_hyperedge<const N: usize>(&self, hyperedge: &[T; N]) -> bool
    where
        T: Hash + Eq,
        Self: HypergraphAccessor<N, T, W>,
    {
        self.edges().contains(hyperedge)
    }

    pub fn get_hyperedge<const N: usize>(&self, hyperedge: &[T; N]) -> Option<&Hx<N, T, W>>
    where
        T: Hash + Eq,
        Self: HypergraphAccessor<N, T, W>,
    {
        self.edges().get(hyperedge)
    }

    pub fn remove_hyperedge<const N: usize>(&mut self, hyperedge: &[T; N]) -> bool
    where
        T: Hash + Eq,
        Self: HypergraphAccessor<N, T, W>,
    {
        self.edges_mut().remove(hyperedge)
    }

    pub fn take_hyperedge<const N: usize>(&mut self, hyperedge: &[T; N]) -> Option<Hx<N, T, W>>
    where
        T: Hash + Eq,
        Self: HypergraphAccessor<N, T, W>,
    {
        self.edges_mut().take(hyperedge)
    }

    pub fn iter_edges<const N: usize>(&self) -> impl Iterator<Item = &Hx<N, T, W>>
    where
        T: Hash + Eq,
        Self: HypergraphAccessor<N, T, W>,
    {
        self.edges().iter()
    }

    pub fn into_iter_edges<const N: usize>(&mut self) -> impl Iterator<Item = Hx<N, T, W>>
    where
        T: Hash + Eq,
        Self: HypergraphAccessor<N, T, W>,
    {
        self.take_edges().into_iter()
    }

    pub fn remove_isolated_nodes(&mut self) -> usize {
        let len = self.nodes.len();
        self.nodes.retain(|_, &mut degree| degree > 0);
        self.n = self.nodes.len();
        len - self.nodes.len()
    }
}

#[pyclass]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
pub struct WeightedHypergraph(pub Hypergraph<NodeId, NodeWeight>);
pub struct WeightedHx<const N: usize>(pub Hx<N, NodeId, NodeWeight>);

#[pyclass]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
pub struct UnweightedHypergraph(pub Hypergraph<NodeId, ()>);
pub struct UnweightedHx<const N: usize>(pub Hx<N, NodeId, ()>);

impl<const N: usize> WeightedHx<N> {
    pub fn new(nodes: [NodeId; N], weight: NodeWeight) -> Result<Self, GraphError<NodeId>> {
        let inner = Hx::new(nodes, weight)?;
        Ok(Self(inner))
    }
    pub fn new_unchecked(nodes: [NodeId; N], weight: NodeWeight) -> Self {
        let inner = Hx::new_unchecked(nodes, weight);
        Self(inner)
    }
}

impl<const N: usize> UnweightedHx<N> {
    pub fn new(nodes: [NodeId; N]) -> Result<Self, GraphError<NodeId>> {
        let inner = Hx::new_unweighted(nodes)?;
        Ok(Self(inner))
    }
    pub fn new_unchecked(nodes: [NodeId; N]) -> Self {
        let inner = Hx::new_unweighted_unchecked(nodes);
        Self(inner)
    }
}

impl<const N: usize> Into<WeightedHx<N>> for Hx<N, NodeId, NodeWeight> {
    fn into(self) -> WeightedHx<N> {
        WeightedHx(self)
    }
}

impl<const N: usize> Into<UnweightedHx<N>> for Hx<N, NodeId, ()> {
    fn into(self) -> UnweightedHx<N> {
        UnweightedHx(self)
    }
}

// Common methods for both weighted and unweighted hypergraphs
#[duplicate_item(
    hg_type                 hx_type;
    [UnweightedHypergraph]  [UnweightedHx];
    [WeightedHypergraph]    [WeightedHx];
)]
#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl hg_type {
    #[new]
    pub fn new() -> Self {
        Self(Hypergraph::new())
    }

    #[staticmethod]
    pub fn from_edges(edges: Vec<Bound<'_, PyTuple>>) -> Self {
        let mut hx = Self(Hypergraph::new());
        for edge in edges {
            hx.insert_hx_tuple(edge).unwrap();
        }
        hx
    }

    pub fn n(&self) -> usize {
        self.0.n
    }

    pub fn m(&self) -> usize {
        self.0.m
    }

    pub fn nodes(&self) -> HashMap<NodeId, usize> {
        self.0.nodes.clone()
    }

    pub fn count(&self, order: usize) -> PyResult<usize> {
        seq!( N in 2..11 {
        match order {
             #( N => Ok(self.0.edges::<N>().len()),             )*
                _ => Err(GraphError::UnsupportedHyperedgeSize(order).into()),
            }
        })
    }

    pub fn edges_by_order<'a>(
        &self,
        py: Python<'a>,
        order: usize,
    ) -> PyResult<Vec<Bound<'a, PyTuple>>> {
        seq!( N in 2..11 {
        match order {
             #( N => Ok(self
                        .0
                        .iter_edges::<N>()
                        .cloned()
                        .map(|e| {
                            let wrapped:hx_type<N> = e.into();
                            wrapped.into_pyobject(py).unwrap()
                        })
                        .collect()),
              )*
                _ => Err(GraphError::UnsupportedHyperedgeSize(order).into()),
            }
        })
    }

    pub fn edges<'a>(&self, py: Python<'a>) -> Vec<Bound<'a, PyTuple>> {
        let mut rv = Vec::new();
        seq!( N in 2..11 {
            rv.extend(self.edges_by_order(py, N).unwrap());
        });
        rv
    }

    fn has_hx(&self, edge: Bound<'_, PyTuple>) -> PyResult<bool> {
        let order = edge.len()?;
        seq!( N in 2..11 {
        match order {
            #(
                N => {
                    let hx: UnweightedHx<N> = edge.extract()?;
                    Ok(self.0.has_hyperedge(&hx.0.nodes))
                },
            )*
                _ => Err(GraphError::UnsupportedHyperedgeSize(order).into()),
        }
        })
    }

    fn get_hx<'a>(
        &self,
        py: Python<'a>,
        edge: Bound<'_, PyTuple>,
    ) -> PyResult<Option<Bound<'a, PyTuple>>> {
        let order = edge.len()?;
        seq!( N in 2..11 {
        match order {
            #(
                N => {
                    let hx: UnweightedHx<N> = edge.extract()?;
                    Ok(self
                        .0
                        .get_hyperedge(&hx.0.nodes)
                        .map(|e| {
                            let wrapped:hx_type<N> = e.clone().into();
                            wrapped.into_pyobject(py).unwrap()
                        }
                        ))
                },
            )*
                _ => Err(GraphError::UnsupportedHyperedgeSize(order).into()),
        }
        })
    }

    fn remove_hx(&mut self, edge: Bound<'_, PyTuple>) -> PyResult<bool> {
        let order = edge.len()?;
        seq!( N in 2..11 {
        match order {
            #(
                N => {
                    let nodes = edge.extract::<UnweightedHx<N>>()?.0.nodes;
                    Ok(self.0.remove_edge_unchecked(&nodes))
                },
            )*
                _ => Err(GraphError::UnsupportedHyperedgeSize(order).into()),
        }
        })
    }

    fn remove_isolated_nodes(&mut self) -> usize {
        self.0.remove_isolated_nodes()
    }
}

// Methods requiring specific implementation for weighted and unweighted hypergraphs
#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl UnweightedHypergraph {
    fn insert_hx(&mut self, edge: Bound<'_, PyTuple>) -> PyResult<bool> {
        let order = edge.len()?;
        seq!(N in 2..11 {
        match order {
            #(
                N => {
                    let unweighted_hx: UnweightedHx<N> = edge.extract()?;
                    Ok(self.0.add_edge(unweighted_hx.0))
                },
            )*
                _ => Err(GraphError::UnsupportedHyperedgeSize(order).into()),
        }
        })
    }

    fn insert_hx_tuple(&mut self, edge: Bound<'_, PyTuple>) -> PyResult<bool> {
        let order = edge.len()?;

        seq!(N in 2..11 {
        match order {
            #(
                N => {
                    let hx: UnweightedHx<N> = edge.extract()?;
                    Ok(self.0.add_edge(hx.0))
                },
            )*
                _ => Err(GraphError::UnsupportedHyperedgeSize(order).into()),
        }
        })
    }
}

// Methods requiring specific implementation for weighted and unweighted hypergraphs
#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl WeightedHypergraph {
    fn insert_hx(&mut self, edge: Bound<'_, PyTuple>, weight: NodeWeight) -> PyResult<bool> {
        let order = edge.len()?;
        seq!(N in 2..11 {
        match order {
            #(
                N => {
                    let unweighted_hx: UnweightedHx<N> = edge.extract()?;
                    let weighted_hx = WeightedHx::new_unchecked(unweighted_hx.0.nodes, weight);
                    Ok(self.0.add_edge(weighted_hx.0))
                },
            )*
                _ => Err(GraphError::UnsupportedHyperedgeSize(order).into()),
        }
        })
    }

    fn insert_hx_tuple(&mut self, edge: Bound<'_, PyTuple>) -> PyResult<bool> {
        let len = edge.len()?;

        if len != 2 {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Expected a tuple of length 2 (nodes, weight), got {}",
                len
            )));
        }

        let first_item = edge.get_item(0)?;
        let nodes = first_item.cast::<PyTuple>()?;
        let weight = edge.get_item(1)?.extract::<NodeWeight>()?;
        let order = nodes.len()?;

        seq!(N in 2..11 {
        match order {
            #(
                N => {
                    let mut nodes_v = [0; N];
                    for i in 0..N {
                        nodes_v[i] = nodes.get_item(i)?.extract::<NodeId>()?;
                    }
                    let hx: WeightedHx<N> = WeightedHx::new(nodes_v, weight)?;
                    Ok(self.0.add_edge(hx.0))
                },
            )*
                _ => Err(GraphError::UnsupportedHyperedgeSize(order).into()),
        }
        })
    }
}
