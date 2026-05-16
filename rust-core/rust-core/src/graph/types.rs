use duplicate::{duplicate, duplicate_item};
use rust_core_macros::hoist_mod;
use seq_macro::seq;
use std::{hash::Hash, ops::Index};

use pyo3::{
    Bound, FromPyObject, IntoPyObject, PyErr, PyResult, Python, pyclass, pymethods,
    types::{PyAnyMethods, PyTuple},
};
use pyo3_stub_gen::{
    derive::{gen_methods_from_python, gen_stub_pyclass, gen_stub_pymethods},
    impl_stub_type,
    inventory::submit,
};
use rkyv::{Archive, Archived, Deserialize, Serialize};

use super::error::GraphError;

pub type NodeId = u32;
pub type NodeWeight = f32;

pub trait Hyperedge<const N: usize> {
    const ORDER: usize;
    type NodeIdType: Archive;
    fn nodes(&self) -> &[Self::NodeIdType; N];
    fn nodes_mut(&mut self) -> &mut [Self::NodeIdType; N];
    fn drain_nodes(self) -> [Self::NodeIdType; N];
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
pub struct Hx<const N: usize, T> {
    nodes: [T; N],
}

impl<const N: usize> Hx<N, NodeId> {
    pub fn new(mut nodes: [NodeId; N]) -> Result<Self, GraphError> {
        nodes.sort_unstable();
        nodes
            .windows(2)
            .find(|w| w[0] == w[1])
            .map(|w| w[0])
            .map_or_else(
                || Ok(Self { nodes }),
                |duplicate| Err(GraphError::DuplicateNodes(duplicate)),
            )
    }

    pub fn new_unchecked(mut nodes: [NodeId; N]) -> Self {
        Self { nodes }
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
pub struct WHx<const N: usize, T> {
    nodes: [T; N],
    weight: NodeWeight,
}

impl<const N: usize> WHx<N, NodeId> {
    pub fn new(weight: NodeWeight, nodes: [NodeId; N]) -> Result<Self, GraphError> {
        let hx = Hx::new(nodes)?;
        Ok(Self {
            nodes: hx.nodes,
            weight,
        })
    }

    pub fn new_unchecked(weight: NodeWeight, nodes: [NodeId; N]) -> Self {
        Self { nodes, weight }
    }
}

#[hoist_mod(attr(duplicate_item(
    hx_type         node_type;
    [Hx]            [NodeId];
    [WHx]           [NodeId];
    [ArchivedHx]    [<NodeId as Archive>::Archived];
    [ArchivedWHx]   [<NodeId as Archive>::Archived];
)))]
mod __ {
    impl<const N: usize> Hyperedge<N> for hx_type<N, node_type> {
        type NodeIdType = node_type;
        const ORDER: usize = N;
        fn nodes(&self) -> &[Self::NodeIdType; N] {
            &self.nodes
        }

        fn nodes_mut(&mut self) -> &mut [Self::NodeIdType; N] {
            &mut self.nodes
        }

        fn drain_nodes(self) -> [Self::NodeIdType; N] {
            self.nodes
        }
    }
}

#[hoist_mod(attr(duplicate_item(hx_type; [Hx]; [WHx])))]
mod __ {

    impl<const N: usize, T> Hash for hx_type<N, T>
    where
        T: Hash,
    {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.nodes.hash(state);
        }
    }

    impl<const N: usize, T> PartialEq for hx_type<N, T>
    where
        T: PartialEq,
    {
        fn eq(&self, other: &Self) -> bool {
            self.nodes == other.nodes
        }
    }

    impl<const N: usize, T> Eq for hx_type<N, T> where T: Eq {}

    // Into Iter
    impl<const N: usize, T> IntoIterator for hx_type<N, T> {
        type Item = T;
        type IntoIter = std::array::IntoIter<T, N>;

        fn into_iter(self) -> Self::IntoIter {
            self.nodes.into_iter()
        }
    }

    // Iter
    impl<'a, const N: usize, T> IntoIterator for &'a hx_type<N, T> {
        type Item = &'a T;
        type IntoIter = std::slice::Iter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.nodes.iter()
        }
    }

    // IterMut
    impl<'a, const N: usize, T> IntoIterator for &'a mut hx_type<N, T> {
        type Item = &'a mut T;
        type IntoIter = std::slice::IterMut<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.nodes.iter_mut()
        }
    }
}

#[hoist_mod(attr(duplicate_item(hx_type; [ArchivedHx]; [ArchivedWHx])))]
mod __ {

    impl<const N: usize, T> PartialEq for hx_type<N, T>
    where
        T: Archive,
        <T as Archive>::Archived: PartialEq,
    {
        fn eq(&self, other: &Self) -> bool {
            self.nodes == other.nodes
        }
    }

    impl<const N: usize, T> Eq for hx_type<N, T>
    where
        T: Archive,
        <T as Archive>::Archived: PartialEq,
    {
    }

    impl<const N: usize, T> Hash for hx_type<N, T>
    where
        T: Archive,
        <T as Archive>::Archived: Hash,
    {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.nodes.hash(state);
        }
    }

    // Into Iter
    impl<const N: usize, T> IntoIterator for hx_type<N, T>
    where
        T: Archive,
    {
        type Item = <T as Archive>::Archived;
        type IntoIter = std::array::IntoIter<<T as Archive>::Archived, N>;

        fn into_iter(self) -> Self::IntoIter {
            self.nodes.into_iter()
        }
    }

    // Iter
    impl<'a, const N: usize, T> IntoIterator for &'a hx_type<N, T>
    where
        T: Archive,
    {
        type Item = &'a <T as Archive>::Archived;
        type IntoIter = std::slice::Iter<'a, <T as Archive>::Archived>;

        fn into_iter(self) -> Self::IntoIter {
            self.nodes.iter()
        }
    }

    // IterMut
    impl<'a, const N: usize, T> IntoIterator for &'a mut hx_type<N, T>
    where
        T: Archive,
    {
        type Item = &'a mut <T as Archive>::Archived;
        type IntoIter = std::slice::IterMut<'a, <T as Archive>::Archived>;

        fn into_iter(self) -> Self::IntoIter {
            self.nodes.iter_mut()
        }
    }
}

impl<'py, const N: usize> IntoPyObject<'py> for Hx<N, NodeId> {
    type Target = PyTuple;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyTuple::new(py, self.nodes()).unwrap())
    }
}

#[duplicate_item(hx_size; [2]; [3]; [4]; [5];)]
impl<'py> FromPyObject<'_, 'py> for Hx<hx_size, NodeId> {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'_, 'py, pyo3::PyAny>) -> Result<Self, Self::Error> {
        let tuple = obj.cast::<PyTuple>()?;

        let order = tuple.len()?;
        if order != hx_size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected a tuple of length {}, got {}",
                hx_size, order
            )));
        }

        let mut v: [NodeId; hx_size] = [0; hx_size];
        for i in 0..hx_size {
            v[i] = tuple.get_item(i)?.extract::<NodeId>()?
        }

        Ok(Self::new(v)?)
    }
}

// duplicate! {
// [
//     hx_type         node_type;
//     [Hx]            [NodeId];
//     [WHx]           [NodeId];
//     // [ArchivedHx]    [<NodeId as Archive>::Archived];
//     // [ArchivedWHx]   [<NodeId as Archive>::Archived];
// ]
//
//
//
// }

// #[duplicate_item(hx_type node_type;
//     [Hx]            [NodeId];
//     [WHx]           [NodeId];
//     [ArchivedHx]    [<NodeId as Archive>::Archived];
//     [ArchivedWHx]   [<NodeId as Archive>::Archived];
// )]
// #[duplicate_item(hx_type; [Hx]; [WHx]; [ArchivedHx]; [ArchivedWHx];)]
//
//
// #[duplicate_item(hx_type; [Hx]; [WHx];
//     // [ArchivedHx]; [ArchivedWHx];
// )]
//
//
// #[duplicate_item(hx_type; [Hx]; [WHx]; [ArchivedHx]; [ArchivedWHx];)]
// impl<const N: usize> PartialEq for hx_type<N> {
//     fn eq(&self, other: &Self) -> bool {
//         self.nodes() == other.nodes()
//     }
// }
//
// #[duplicate_item(hx_type; [Hx]; [WHx]; [ArchivedHx]; [ArchivedWHx];)]
// impl<const N: usize> Eq for hx_type<N> {}
//
//
// #[duplicate_item(hx_type node_type;
//     [Hx]            [NodeId];
//     [WHx]           [NodeId];
//     [ArchivedHx]    [<NodeId as Archive>::Archived];
//     // [ArchivedWHx]   [<NodeId as Archive>::Archived];
// )]
// impl<const N: usize> IntoIterator for hx_type<N> {
//     type Item = NodeId;
//     type IntoIter = std::array::IntoIter<<Self as Hyperedge<N>>::NodeIdType, N>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         self.drain_nodes().into_iter()
//     }
// }

// Iter
// impl<'a, const N: usize> IntoIterator for &'a hx_type<N> {
//     type Item = &'a NodeId;
//     type IntoIter = std::slice::Iter<'a, NodeId>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         self.nodes.iter()
//     }
// }

// IterMut
// impl<'a, const N: usize> IntoIterator for &'a mut hx_type<N> {
//     type Item = &'a mut NodeId;
//     type IntoIter = std::slice::IterMut<'a, NodeId>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         self.nodes.iter_mut()
//     }
// }

// #[duplicate_item(hx_type; [Hx]; [WHx];)]
// impl<const N: usize> Hash for hx_type<N> {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         self.nodes.hash(state);
//     }
// }
// #[duplicate_item(hx_type; [Hx]; [WHx];)]
// impl<const N: usize> PartialEq for hx_type<N> {
//     fn eq(&self, other: &Self) -> bool {
//         self.nodes == other.nodes
//     }
// }
// #[duplicate_item(hx_type; [Hx]; [WHx];)]
// impl<const N: usize> Eq for hx_type<N> {}
//
// #[duplicate_item(hx_type; [ArchivedHx]; [ArchivedWHx];)]
// impl<const N: usize> Hash for hx_type<N> {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         self.nodes.hash(state);
//     }
// }
// #[duplicate_item(hx_type; [ArchivedHx]; [ArchivedWHx];)]
// impl<const N: usize> PartialEq for hx_type<N> {
//     fn eq(&self, other: &Self) -> bool {
//         self.nodes == other.nodes
//     }
// }
// #[duplicate_item(hx_type; [ArchivedHx]; [ArchivedWHx];)]
// impl<const N: usize> Eq for hx_type<N> {}

// IntoIter

// --- Unweighted Hyperedges ---
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
// #[pyclass]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// #[rkyv(derive(PartialEq), derive(Eq), derive(Hash))]
// pub struct H2(pub NodeId, pub NodeId);
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl H2 {
//     #[new]
//     #[gen_stub(skip)]
//     pub fn new(u: NodeId, v: NodeId) -> PyResult<Self> {
//         if u == v {
//             return Err(GraphError::DuplicateNodes(u).into());
//         }
//         Ok(if u < v { Self(u, v) } else { Self(v, u) })
//     }
//
//     #[staticmethod]
//     pub fn new_unchecked(u: NodeId, v: NodeId) -> Self {
//         Self(u, v)
//     }
// }
//
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
// #[pyclass]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// pub struct H3(pub NodeId, pub NodeId, pub NodeId);
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl H3 {
//     #[new]
//     #[gen_stub(skip)]
//     pub fn new(u: NodeId, v: NodeId, w: NodeId) -> PyResult<Self> {
//         let mut nodes = [u, v, w];
//         nodes.sort_unstable();
//         if let Some(pair) = nodes.windows(2).find(|w| w[0] == w[1]) {
//             return Err(GraphError::DuplicateNodes(pair[0]).into());
//         }
//         Ok(Self(nodes[0], nodes[1], nodes[2]))
//     }
//
//     #[staticmethod]
//     pub fn new_unchecked(u: NodeId, v: NodeId, w: NodeId) -> Self {
//         Self(u, v, w)
//     }
// }
//
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
// #[pyclass]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// pub struct H4(pub NodeId, pub NodeId, pub NodeId, pub NodeId);
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl H4 {
//     #[new]
//     #[gen_stub(skip)]
//     pub fn new(u: NodeId, v: NodeId, w: NodeId, x: NodeId) -> PyResult<Self> {
//         let mut nodes = [u, v, w, x];
//         nodes.sort_unstable();
//         if let Some(pair) = nodes.windows(2).find(|w| w[0] == w[1]) {
//             return Err(GraphError::DuplicateNodes(pair[0]).into());
//         }
//         Ok(Self(nodes[0], nodes[1], nodes[2], nodes[3]))
//     }
//
//     #[staticmethod]
//     pub fn new_unchecked(u: NodeId, v: NodeId, w: NodeId, x: NodeId) -> Self {
//         Self(u, v, w, x)
//     }
// }
//
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
// #[pyclass]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// pub struct H5(pub NodeId, pub NodeId, pub NodeId, pub NodeId, pub NodeId);
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl H5 {
//     #[new]
//     #[gen_stub(skip)]
//     pub fn new(u: NodeId, v: NodeId, w: NodeId, x: NodeId, y: NodeId) -> PyResult<Self> {
//         let mut nodes = [u, v, w, x, y];
//         nodes.sort_unstable();
//         if let Some(pair) = nodes.windows(2).find(|w| w[0] == w[1]) {
//             return Err(GraphError::DuplicateNodes(pair[0]).into());
//         }
//         Ok(Self(nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]))
//     }
//
//     #[staticmethod]
//     pub fn new_unchecked(u: NodeId, v: NodeId, w: NodeId, x: NodeId, y: NodeId) -> Self {
//         Self(u, v, w, x, y)
//     }
// }
//
// // --- Weighted Hyperedges ---
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
// #[pyclass]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// pub struct WH2 {
//     pub weight: NodeWeight,
//     pub nodes: H2,
// }
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl WH2 {
//     #[new]
//     #[gen_stub(skip)]
//     pub fn new(weight: NodeWeight, u: NodeId, v: NodeId) -> PyResult<Self> {
//         Ok(WH2 {
//             weight,
//             nodes: H2::new(u, v)?,
//         })
//     }
//
//     #[staticmethod]
//     pub fn new_unchecked(weight: NodeWeight, u: NodeId, v: NodeId) -> Self {
//         Self {
//             weight,
//             nodes: H2::new_unchecked(u, v),
//         }
//     }
// }
//
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
// #[pyclass]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// pub struct WH3 {
//     pub weight: NodeWeight,
//     pub nodes: H3,
// }
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl WH3 {
//     #[new]
//     #[gen_stub(skip)]
//     pub fn new(weight: NodeWeight, u: NodeId, v: NodeId, w: NodeId) -> PyResult<Self> {
//         Ok(WH3 {
//             weight,
//             nodes: H3::new(u, v, w)?,
//         })
//     }
//     #[staticmethod]
//     pub fn new_unchecked(weight: NodeWeight, u: NodeId, v: NodeId, w: NodeId) -> Self {
//         Self {
//             weight,
//             nodes: H3::new_unchecked(u, v, w),
//         }
//     }
// }
//
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
// #[pyclass]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// pub struct WH4 {
//     pub weight: NodeWeight,
//     pub nodes: H4,
// }
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl WH4 {
//     #[new]
//     #[gen_stub(skip)]
//     pub fn new(weight: NodeWeight, u: NodeId, v: NodeId, w: NodeId, x: NodeId) -> PyResult<Self> {
//         Ok(WH4 {
//             weight,
//             nodes: H4::new(u, v, w, x)?,
//         })
//     }
//
//     #[staticmethod]
//     pub fn new_unchecked(weight: NodeWeight, u: NodeId, v: NodeId, w: NodeId, x: NodeId) -> Self {
//         Self {
//             weight,
//             nodes: H4::new_unchecked(u, v, w, x),
//         }
//     }
// }
//
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
// #[pyclass]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// pub struct WH5 {
//     pub weight: NodeWeight,
//     pub nodes: H5,
// }
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl WH5 {
//     #[new]
//     #[gen_stub(skip)]
//     pub fn new(
//         weight: NodeWeight,
//         u: NodeId,
//         v: NodeId,
//         w: NodeId,
//         x: NodeId,
//         y: NodeId,
//     ) -> PyResult<Self> {
//         Ok(WH5 {
//             weight,
//             nodes: H5::new(u, v, w, x, y)?,
//         })
//     }
//
//     #[staticmethod]
//     pub fn new_unchecked(
//         weight: NodeWeight,
//         u: NodeId,
//         v: NodeId,
//         w: NodeId,
//         x: NodeId,
//         y: NodeId,
//     ) -> Self {
//         Self {
//             weight,
//             nodes: H5::new_unchecked(u, v, w, x, y),
//         }
//     }
// }
//
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, Eq, Hash)]
// #[rkyv(derive(Debug), derive(PartialEq), derive(Eq), derive(Hash))]
// #[pyclass]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// pub struct Hx {
//     pub nodes: Vec<NodeId>,
// }
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl Hx {
//     #[new]
//     pub fn new(nodes: Vec<NodeId>) -> PyResult<Self> {
//         let mut nodes = nodes;
//         nodes.sort_unstable();
//         if let Some(pair) = nodes.windows(2).find(|w| w[0] == w[1]) {
//             return Err(GraphError::DuplicateNodes(pair[0]).into());
//         }
//         Ok(Self { nodes })
//     }
//
//     #[staticmethod]
//     pub fn new_unchecked(nodes: Vec<NodeId>) -> Self {
//         Self { nodes }
//     }
//
//     pub fn len(&self) -> usize {
//         self.nodes.len()
//     }
// }
//
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
// #[pyclass]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// pub struct WHx {
//     pub weight: NodeWeight,
//     pub edge: Hx,
// }
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl WHx {
//     #[new]
//     pub fn new(weight: NodeWeight, nodes: Vec<NodeId>) -> PyResult<Self> {
//         Ok(Self {
//             weight,
//             edge: Hx::new(nodes)?,
//         })
//     }
//
//     #[staticmethod]
//     pub fn new_unchecked(weight: NodeWeight, nodes: Vec<NodeId>) -> Self {
//         Self {
//             weight,
//             edge: Hx::new_unchecked(nodes),
//         }
//     }
//
//     pub fn len(&self) -> usize {
//         self.edge.len()
//     }
// }
//
// // -- From for Hx
// impl From<H2> for Hx {
//     fn from(h: H2) -> Self {
//         Self {
//             nodes: vec![h.0, h.1],
//         }
//     }
// }
//
// impl From<H3> for Hx {
//     fn from(h: H3) -> Self {
//         Self {
//             nodes: vec![h.0, h.1, h.2],
//         }
//     }
// }
//
// impl From<H4> for Hx {
//     fn from(h: H4) -> Self {
//         Self {
//             nodes: vec![h.0, h.1, h.2, h.3],
//         }
//     }
// }
//
// impl From<H5> for Hx {
//     fn from(h: H5) -> Self {
//         Self {
//             nodes: vec![h.0, h.1, h.2, h.3, h.4],
//         }
//     }
// }
//
// // -- From for WHx
// impl From<WH2> for WHx {
//     fn from(h: WH2) -> Self {
//         Self {
//             weight: h.weight,
//             edge: Hx::new_unchecked(vec![h.nodes.0, h.nodes.1]),
//         }
//     }
// }
//
// impl From<WH3> for WHx {
//     fn from(h: WH3) -> Self {
//         Self {
//             weight: h.weight,
//             edge: Hx::new_unchecked(vec![h.nodes.0, h.nodes.1, h.nodes.2]),
//         }
//     }
// }
//
// impl From<WH4> for WHx {
//     fn from(h: WH4) -> Self {
//         Self {
//             weight: h.weight,
//             edge: Hx::new_unchecked(vec![h.nodes.0, h.nodes.1, h.nodes.2, h.nodes.3]),
//         }
//     }
// }
//
// impl From<WH5> for WHx {
//     fn from(h: WH5) -> Self {
//         Self {
//             weight: h.weight,
//             edge: Hx::new_unchecked(vec![h.nodes.0, h.nodes.1, h.nodes.2, h.nodes.3, h.nodes.4]),
//         }
//     }
// }
//
// // -- From for tuple specific hyperedges
// impl TryFrom<Hx> for H2 {
//     type Error = GraphError;
//     fn try_from(h: Hx) -> Result<Self, Self::Error> {
//         if h.nodes.len() != 2 {
//             return Err(GraphError::InvalidHyperedgeSize(h.len(), 2));
//         }
//         Ok(Self(h.nodes[0], h.nodes[1]))
//     }
// }
//
// impl TryFrom<Hx> for H3 {
//     type Error = GraphError;
//     fn try_from(h: Hx) -> Result<Self, Self::Error> {
//         if h.nodes.len() != 3 {
//             return Err(GraphError::InvalidHyperedgeSize(h.len(), 3));
//         }
//         Ok(Self(h.nodes[0], h.nodes[1], h.nodes[2]))
//     }
// }
//
// impl TryFrom<Hx> for H4 {
//     type Error = GraphError;
//     fn try_from(h: Hx) -> Result<Self, Self::Error> {
//         if h.nodes.len() != 4 {
//             return Err(GraphError::InvalidHyperedgeSize(h.len(), 4));
//         }
//         Ok(Self(h.nodes[0], h.nodes[1], h.nodes[2], h.nodes[3]))
//     }
// }
//
// impl TryFrom<Hx> for H5 {
//     type Error = GraphError;
//     fn try_from(h: Hx) -> Result<Self, Self::Error> {
//         if h.nodes.len() != 5 {
//             return Err(GraphError::InvalidHyperedgeSize(h.len(), 5));
//         }
//         Ok(Self(
//             h.nodes[0], h.nodes[1], h.nodes[2], h.nodes[3], h.nodes[4],
//         ))
//     }
// }
//
// impl Index<usize> for Hx {
//     type Output = NodeId;
//
//     fn index(&self, index: usize) -> &Self::Output {
//         &self.nodes[index]
//     }
// }
//
// impl Index<usize> for WHx {
//     type Output = NodeId;
//
//     fn index(&self, index: usize) -> &Self::Output {
//         &self.edge[index]
//     }
// }
//
// // -- iter
// impl<'a> IntoIterator for &'a H2 {
//     type Item = &'a NodeId;
//     type IntoIter = std::array::IntoIter<&'a NodeId, 2>;
//
//     #[inline(always)]
//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::IntoIterator::into_iter([&self.0, &self.1])
//     }
// }
//
// impl<'a> IntoIterator for &'a H3 {
//     type Item = &'a NodeId;
//     type IntoIter = std::array::IntoIter<&'a NodeId, 3>;
//
//     #[inline(always)]
//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::IntoIterator::into_iter([&self.0, &self.1, &self.2])
//     }
// }
//
// impl<'a> IntoIterator for &'a H4 {
//     type Item = &'a NodeId;
//     type IntoIter = std::array::IntoIter<&'a NodeId, 4>;
//
//     #[inline(always)]
//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::IntoIterator::into_iter([&self.0, &self.1, &self.2, &self.3])
//     }
// }
//
// impl<'a> IntoIterator for &'a H5 {
//     type Item = &'a NodeId;
//     type IntoIter = std::array::IntoIter<&'a NodeId, 5>;
//
//     #[inline(always)]
//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::IntoIterator::into_iter([&self.0, &self.1, &self.2, &self.3, &self.4])
//     }
// }
//
// impl<'a> IntoIterator for &'a Hx {
//     type Item = &'a NodeId;
//     type IntoIter = std::slice::Iter<'a, NodeId>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         self.nodes.iter()
//     }
// }
//
// impl<'a> IntoIterator for &'a WHx {
//     type Item = &'a NodeId;
//     type IntoIter = std::slice::Iter<'a, NodeId>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         self.edge.nodes.iter()
//     }
// }
//
// // -- into iter
// impl IntoIterator for H2 {
//     type Item = NodeId;
//     type IntoIter = std::array::IntoIter<NodeId, 2>;
//
//     #[inline(always)]
//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::IntoIterator::into_iter([self.0, self.1])
//     }
// }
//
// impl IntoIterator for H3 {
//     type Item = NodeId;
//     type IntoIter = std::array::IntoIter<NodeId, 3>;
//
//     #[inline(always)]
//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::IntoIterator::into_iter([self.0, self.1, self.2])
//     }
// }
//
// impl IntoIterator for H4 {
//     type Item = NodeId;
//     type IntoIter = std::array::IntoIter<NodeId, 4>;
//
//     #[inline(always)]
//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::IntoIterator::into_iter([self.0, self.1, self.2, self.3])
//     }
// }
//
// impl IntoIterator for H5 {
//     type Item = NodeId;
//     type IntoIter = std::array::IntoIter<NodeId, 5>;
//
//     #[inline(always)]
//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::IntoIterator::into_iter([self.0, self.1, self.2, self.3, self.4])
//     }
// }
//
// impl IntoIterator for Hx {
//     type Item = NodeId;
//     type IntoIter = std::vec::IntoIter<NodeId>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         self.nodes.into_iter()
//     }
// }
//
// impl IntoIterator for WHx {
//     type Item = NodeId;
//     type IntoIter = std::vec::IntoIter<NodeId>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         self.edge.into_iter()
//     }
// }
//
// // -- iter mut
// impl<'a> IntoIterator for &'a mut H2 {
//     type Item = &'a mut NodeId;
//     type IntoIter = std::array::IntoIter<&'a mut NodeId, 2>;
//
//     #[inline(always)]
//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::IntoIterator::into_iter([&mut self.0, &mut self.1])
//     }
// }
//
// impl<'a> IntoIterator for &'a mut H3 {
//     type Item = &'a mut NodeId;
//     type IntoIter = std::array::IntoIter<&'a mut NodeId, 3>;
//
//     #[inline(always)]
//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::IntoIterator::into_iter([&mut self.0, &mut self.1, &mut self.2])
//     }
// }
//
// impl<'a> IntoIterator for &'a mut H4 {
//     type Item = &'a mut NodeId;
//     type IntoIter = std::array::IntoIter<&'a mut NodeId, 4>;
//
//     #[inline(always)]
//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::IntoIterator::into_iter([&mut self.0, &mut self.1, &mut self.2, &mut self.3])
//     }
// }
//
// impl<'a> IntoIterator for &'a mut H5 {
//     type Item = &'a mut NodeId;
//     type IntoIter = std::array::IntoIter<&'a mut NodeId, 5>;
//
//     #[inline(always)]
//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::IntoIterator::into_iter([
//             &mut self.0,
//             &mut self.1,
//             &mut self.2,
//             &mut self.3,
//             &mut self.4,
//         ])
//     }
// }
//
// impl<'a> IntoIterator for &'a mut Hx {
//     type Item = &'a mut NodeId;
//     type IntoIter = std::slice::IterMut<'a, NodeId>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         self.nodes.iter_mut()
//     }
// }
//
// impl<'a> IntoIterator for &'a mut WHx {
//     type Item = &'a mut NodeId;
//     type IntoIter = std::slice::IterMut<'a, NodeId>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         self.edge.nodes.iter_mut()
//     }
// }
//
// #[derive(FromPyObject, Debug)]
// pub enum Hypergraph<'py> {
//     #[pyo3(transparent)]
//     Db(Bound<'py, UnweightedHypergraph>),
//     #[pyo3(transparent)]
//     File(Bound<'py, ArchivedUnweightedHypergraph>),
// }
//
// impl_stub_type!(Hypergraph<'_> = UnweightedHypergraph | ArchivedUnweightedHypergraph);
//
// submit! {
//     gen_methods_from_python! {
//         r#"
//         class H2:
//             def __init__(u: int, v:int) -> H2: ...
//         "#
//     };
//     gen_methods_from_python! {
//         r#"
//         class H3:
//             def __init__(u: int, v:int, w:int) -> H3: ...
//         "#
//     };
//     gen_methods_from_python! {
//         r#"
//         class H4:
//             def __init__(u: int, v:int, w:int) -> H4: ...
//         "#
//     };
//     gen_methods_from_python! {
//         r#"
//         class H5:
//             def __init__(u: int, v:int, w:int) -> H5: ...
//         "#
//     };
//
//     gen_methods_from_python! {
//         r#"
//         class WH2:
//             def __init__(u: int, v:int) -> WH2: ...
//         "#
//     };
//     gen_methods_from_python! {
//         r#"
//         class WH3:
//             def __init__(u: int, v:int, w:int) -> WH3: ...
//         "#
//     };
//     gen_methods_from_python! {
//         r#"
//         class WH4:
//             def __init__(u: int, v:int, w:int) -> WH4: ...
//         "#
//     };
//     gen_methods_from_python! {
//         r#"
//         class WH5:
//             def __init__(u: int, v:int, w:int) -> WH5: ...
//         "#
//     }
// }
