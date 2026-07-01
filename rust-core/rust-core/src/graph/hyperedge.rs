use duplicate::{duplicate, duplicate_item};
use rust_core_macros::hoist_mod;
use seq_macro::seq;
use std::{error::Error, fmt::Display, hash::Hash, ops::Index};

use rkyv::{Archive, Archived, Deserialize, Serialize};

use super::error::GraphError;

pub type NodeId = u32;
pub type EdgeId = u32;
pub type NodeWeight = f32;

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
pub struct Hx<const N: usize, T, W> {
    pub nodes: [T; N],
    pub weight: W,
}

impl<const N: usize, T, W> Hx<N, T, W> {
    pub fn new(mut nodes: [T; N], weight: W) -> Result<Self, GraphError<T>>
    where
        T: Ord + Copy,
    {
        nodes.sort_unstable();

        if let Some(&dup) = nodes
            .windows(2)
            .find_map(|w| (w[0] == w[1]).then_some(&w[0]))
        {
            return Err(GraphError::DuplicateNodes(dup));
        }

        Ok(Self { nodes, weight })
    }

    pub fn new_unchecked(nodes: [T; N], weight: W) -> Self {
        Self { nodes, weight }
    }
}

impl<const N: usize, T> Hx<N, T, ()> {
    pub fn new_unweighted(vertices: [T; N]) -> Result<Self, GraphError<T>>
    where
        T: Ord + Copy,
    {
        Self::new(vertices, ())
    }

    pub fn new_unweighted_unchecked(vertices: [T; N]) -> Self {
        Self::new_unchecked(vertices, ())
    }
}

impl<const N: usize, T, W> Hash for Hx<N, T, W>
where
    T: Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.nodes.hash(state);
    }
}

impl<const N: usize, T, W> PartialEq for Hx<N, T, W>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.nodes == other.nodes
    }
}

impl<const N: usize, T, W> Eq for Hx<N, T, W> where T: Eq {}

// Into Iter
impl<const N: usize, T, W> IntoIterator for Hx<N, T, W> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.into_iter()
    }
}

// Reference Iteration
impl<'a, const N: usize, T, W> IntoIterator for &'a Hx<N, T, W> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.iter()
    }
}

// Mutable Reference Iteration
impl<'a, const N: usize, T, W> IntoIterator for &'a mut Hx<N, T, W> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.iter_mut()
    }
}

impl<'a, T, W, const N: usize> hashbrown::Equivalent<Hx<N, T, W>> for [T; N]
where
    T: PartialEq,
{
    fn equivalent(&self, key: &Hx<N, T, W>) -> bool {
        self == &key.nodes
    }
}

pub struct WeightedHx<const N: usize>(pub Hx<N, NodeId, NodeWeight>);

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

#[cfg(feature = "bindings")]
#[hoist_mod]
mod bindings {
    use pyo3::{
        Bound, FromPyObject, IntoPyObject, PyErr, PyRef, PyResult, Python,
        exceptions::PyValueError,
        pyclass, pymethods,
        types::{PyAny, PyAnyMethods, PySet, PyTuple},
    };

    use pyo3_stub_gen::{
        PyStubType, TypeInfo,
        derive::{gen_stub_pyclass, gen_stub_pymethods},
        impl_stub_type, type_alias,
    };

    impl<'py, const N: usize> IntoPyObject<'py> for UnweightedHx<N> {
        type Target = PyTuple;
        type Output = Bound<'py, Self::Target>;
        type Error = std::convert::Infallible;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            Ok(PyTuple::new(py, self.0.nodes).unwrap())
        }
    }

    impl<'py, const N: usize> IntoPyObject<'py> for WeightedHx<N> {
        type Target = PyTuple;
        type Output = Bound<'py, Self::Target>;
        type Error = std::convert::Infallible;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            Ok((PyTuple::new(py, self.0.nodes).unwrap(), self.0.weight)
                .into_pyobject(py)
                .unwrap())
        }
    }

    impl<'py, 'a, const N: usize> FromPyObject<'py, 'a> for UnweightedHx<N> {
        type Error = PyErr;

        fn extract(obj: pyo3::Borrowed<'py, 'a, PyAny>) -> Result<Self, Self::Error> {
            let tuple = obj.cast::<PyTuple>()?;
            let len = tuple.len()?;

            if len != N {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "Expected a tuple of length 2 (nodes, weight), got {}",
                    len
                )));
            }

            let mut nodes_v = [0; N];
            for i in 0..N {
                nodes_v[i] = tuple.get_item(i)?.extract::<NodeId>()?;
            }

            UnweightedHx::new(nodes_v).map_err(|e| e.into())
        }
    }

    impl<'py, 'a, const N: usize> FromPyObject<'py, 'a> for WeightedHx<N> {
        type Error = PyErr;

        fn extract(obj: pyo3::Borrowed<'py, 'a, PyAny>) -> Result<Self, Self::Error> {
            let tuple = obj.cast::<PyTuple>()?;
            let len = tuple.len()?;

            if len != 2 {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "Expected a tuple of length 2 (nodes, weight), got {}",
                    len
                )));
            }

            let first_item = tuple.get_item(0)?;
            let nodes = first_item.cast::<PyTuple>()?;
            let weight = tuple.get_item(1)?.extract::<NodeWeight>()?;
            let len = nodes.len()?;

            if len != N {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "Expected a tuple of length {}, got {}",
                    N, len
                )));
            }

            let mut nodes_v = [0; N];
            for i in 0..N {
                nodes_v[i] = tuple.get_item(i)?.extract::<NodeId>()?;
            }

            WeightedHx::new(nodes_v, weight).map_err(|e| e.into())
        }
    }
}
