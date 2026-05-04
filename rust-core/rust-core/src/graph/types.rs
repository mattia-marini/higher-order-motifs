use std::ops::Index;

use pyo3::{Bound, FromPyObject, PyResult, pyclass, pymethods};
use pyo3_stub_gen::{
    derive::{gen_methods_from_python, gen_stub_pyclass, gen_stub_pymethods},
    impl_stub_type,
    inventory::submit,
};
use rkyv::{Archive, Deserialize, Serialize};

use crate::graph::{ArchivedUnweightedHypergraph, UnweightedHypergraph};

use super::error::GraphError;

pub type NodeId = u32;
pub type NodeWeight = f32;

// --- Unweighted Hyperedges ---
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
#[pyclass]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
pub struct H2(pub NodeId, pub NodeId);

#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl H2 {
    #[new]
    #[gen_stub(skip)]
    pub fn new(u: NodeId, v: NodeId) -> PyResult<Self> {
        if u == v {
            return Err(GraphError::DuplicateNodes(u).into());
        }
        Ok(if u < v { Self(u, v) } else { Self(v, u) })
    }

    #[staticmethod]
    pub fn new_unchecked(u: NodeId, v: NodeId) -> Self {
        Self(u, v)
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
#[pyclass]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
pub struct H3(pub NodeId, pub NodeId, pub NodeId);

#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl H3 {
    #[new]
    #[gen_stub(skip)]
    pub fn new(u: NodeId, v: NodeId, w: NodeId) -> PyResult<Self> {
        let mut nodes = [u, v, w];
        nodes.sort_unstable();
        if let Some(pair) = nodes.windows(2).find(|w| w[0] == w[1]) {
            return Err(GraphError::DuplicateNodes(pair[0]).into());
        }
        Ok(Self(nodes[0], nodes[1], nodes[2]))
    }

    #[staticmethod]
    pub fn new_unchecked(u: NodeId, v: NodeId, w: NodeId) -> Self {
        Self(u, v, w)
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
#[pyclass]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
pub struct H4(pub NodeId, pub NodeId, pub NodeId, pub NodeId);

#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl H4 {
    #[new]
    #[gen_stub(skip)]
    pub fn new(u: NodeId, v: NodeId, w: NodeId, x: NodeId) -> PyResult<Self> {
        let mut nodes = [u, v, w, x];
        nodes.sort_unstable();
        if let Some(pair) = nodes.windows(2).find(|w| w[0] == w[1]) {
            return Err(GraphError::DuplicateNodes(pair[0]).into());
        }
        Ok(Self(nodes[0], nodes[1], nodes[2], nodes[3]))
    }

    #[staticmethod]
    pub fn new_unchecked(u: NodeId, v: NodeId, w: NodeId, x: NodeId) -> Self {
        Self(u, v, w, x)
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
#[pyclass]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
pub struct H5(pub NodeId, pub NodeId, pub NodeId, pub NodeId, pub NodeId);

#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl H5 {
    #[new]
    #[gen_stub(skip)]
    pub fn new(u: NodeId, v: NodeId, w: NodeId, x: NodeId, y: NodeId) -> PyResult<Self> {
        let mut nodes = [u, v, w, x, y];
        nodes.sort_unstable();
        if let Some(pair) = nodes.windows(2).find(|w| w[0] == w[1]) {
            return Err(GraphError::DuplicateNodes(pair[0]).into());
        }
        Ok(Self(nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]))
    }

    #[staticmethod]
    pub fn new_unchecked(u: NodeId, v: NodeId, w: NodeId, x: NodeId, y: NodeId) -> Self {
        Self(u, v, w, x, y)
    }
}

// --- Weighted Hyperedges ---
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[pyclass]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
pub struct WH2 {
    pub weight: NodeWeight,
    pub nodes: H2,
}

#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl WH2 {
    #[new]
    #[gen_stub(skip)]
    pub fn new(weight: NodeWeight, u: NodeId, v: NodeId) -> PyResult<Self> {
        Ok(WH2 {
            weight,
            nodes: H2::new(u, v)?,
        })
    }

    #[staticmethod]
    pub fn new_unchecked(weight: NodeWeight, u: NodeId, v: NodeId) -> Self {
        Self {
            weight,
            nodes: H2::new_unchecked(u, v),
        }
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[pyclass]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
pub struct WH3 {
    pub weight: NodeWeight,
    pub nodes: H3,
}

#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl WH3 {
    #[new]
    #[gen_stub(skip)]
    pub fn new(weight: NodeWeight, u: NodeId, v: NodeId, w: NodeId) -> PyResult<Self> {
        Ok(WH3 {
            weight,
            nodes: H3::new(u, v, w)?,
        })
    }
    #[staticmethod]
    pub fn new_unchecked(weight: NodeWeight, u: NodeId, v: NodeId, w: NodeId) -> Self {
        Self {
            weight,
            nodes: H3::new_unchecked(u, v, w),
        }
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[pyclass]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
pub struct WH4 {
    pub weight: NodeWeight,
    pub nodes: H4,
}

#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl WH4 {
    #[new]
    #[gen_stub(skip)]
    pub fn new(weight: NodeWeight, u: NodeId, v: NodeId, w: NodeId, x: NodeId) -> PyResult<Self> {
        Ok(WH4 {
            weight,
            nodes: H4::new(u, v, w, x)?,
        })
    }

    #[staticmethod]
    pub fn new_unchecked(weight: NodeWeight, u: NodeId, v: NodeId, w: NodeId, x: NodeId) -> Self {
        Self {
            weight,
            nodes: H4::new_unchecked(u, v, w, x),
        }
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[pyclass]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
pub struct WH5 {
    pub weight: NodeWeight,
    pub nodes: H5,
}

#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl WH5 {
    #[new]
    #[gen_stub(skip)]
    pub fn new(
        weight: NodeWeight,
        u: NodeId,
        v: NodeId,
        w: NodeId,
        x: NodeId,
        y: NodeId,
    ) -> PyResult<Self> {
        Ok(WH5 {
            weight,
            nodes: H5::new(u, v, w, x, y)?,
        })
    }

    #[staticmethod]
    pub fn new_unchecked(
        weight: NodeWeight,
        u: NodeId,
        v: NodeId,
        w: NodeId,
        x: NodeId,
        y: NodeId,
    ) -> Self {
        Self {
            weight,
            nodes: H5::new_unchecked(u, v, w, x, y),
        }
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, Eq, Hash)]
#[rkyv(derive(Debug), derive(PartialEq), derive(Eq), derive(Hash))]
#[pyclass]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
pub struct Hx {
    pub nodes: Vec<NodeId>,
}

#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl Hx {
    #[new]
    pub fn new(nodes: Vec<NodeId>) -> PyResult<Self> {
        let mut nodes = nodes;
        nodes.sort_unstable();
        if let Some(pair) = nodes.windows(2).find(|w| w[0] == w[1]) {
            return Err(GraphError::DuplicateNodes(pair[0]).into());
        }
        Ok(Self { nodes })
    }

    #[staticmethod]
    pub fn new_unchecked(nodes: Vec<NodeId>) -> Self {
        Self { nodes }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[pyclass]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
pub struct WHx {
    pub weight: NodeWeight,
    pub edge: Hx,
}

#[pymethods]
#[gen_stub_pymethods(module = "rust_core.core.graph")]
impl WHx {
    #[new]
    pub fn new(weight: NodeWeight, nodes: Vec<NodeId>) -> PyResult<Self> {
        Ok(Self {
            weight,
            edge: Hx::new(nodes)?,
        })
    }

    #[staticmethod]
    pub fn new_unchecked(weight: NodeWeight, nodes: Vec<NodeId>) -> Self {
        Self {
            weight,
            edge: Hx::new_unchecked(nodes),
        }
    }

    pub fn len(&self) -> usize {
        self.edge.len()
    }
}

// -- From for Hx
impl From<H2> for Hx {
    fn from(h: H2) -> Self {
        Self {
            nodes: vec![h.0, h.1],
        }
    }
}

impl From<H3> for Hx {
    fn from(h: H3) -> Self {
        Self {
            nodes: vec![h.0, h.1, h.2],
        }
    }
}

impl From<H4> for Hx {
    fn from(h: H4) -> Self {
        Self {
            nodes: vec![h.0, h.1, h.2, h.3],
        }
    }
}

impl From<H5> for Hx {
    fn from(h: H5) -> Self {
        Self {
            nodes: vec![h.0, h.1, h.2, h.3, h.4],
        }
    }
}

// -- From for WHx
impl From<WH2> for WHx {
    fn from(h: WH2) -> Self {
        Self {
            weight: h.weight,
            edge: Hx::new_unchecked(vec![h.nodes.0, h.nodes.1]),
        }
    }
}

impl From<WH3> for WHx {
    fn from(h: WH3) -> Self {
        Self {
            weight: h.weight,
            edge: Hx::new_unchecked(vec![h.nodes.0, h.nodes.1, h.nodes.2]),
        }
    }
}

impl From<WH4> for WHx {
    fn from(h: WH4) -> Self {
        Self {
            weight: h.weight,
            edge: Hx::new_unchecked(vec![h.nodes.0, h.nodes.1, h.nodes.2, h.nodes.3]),
        }
    }
}

impl From<WH5> for WHx {
    fn from(h: WH5) -> Self {
        Self {
            weight: h.weight,
            edge: Hx::new_unchecked(vec![h.nodes.0, h.nodes.1, h.nodes.2, h.nodes.3, h.nodes.4]),
        }
    }
}

// -- From for tuple specific hyperedges
impl TryFrom<Hx> for H2 {
    type Error = GraphError;
    fn try_from(h: Hx) -> Result<Self, Self::Error> {
        if h.nodes.len() != 2 {
            return Err(GraphError::InvalidHyperedgeSize(h.len(), 2));
        }
        Ok(Self(h.nodes[0], h.nodes[1]))
    }
}

impl TryFrom<Hx> for H3 {
    type Error = GraphError;
    fn try_from(h: Hx) -> Result<Self, Self::Error> {
        if h.nodes.len() != 3 {
            return Err(GraphError::InvalidHyperedgeSize(h.len(), 3));
        }
        Ok(Self(h.nodes[0], h.nodes[1], h.nodes[2]))
    }
}

impl TryFrom<Hx> for H4 {
    type Error = GraphError;
    fn try_from(h: Hx) -> Result<Self, Self::Error> {
        if h.nodes.len() != 4 {
            return Err(GraphError::InvalidHyperedgeSize(h.len(), 4));
        }
        Ok(Self(h.nodes[0], h.nodes[1], h.nodes[2], h.nodes[3]))
    }
}

impl TryFrom<Hx> for H5 {
    type Error = GraphError;
    fn try_from(h: Hx) -> Result<Self, Self::Error> {
        if h.nodes.len() != 5 {
            return Err(GraphError::InvalidHyperedgeSize(h.len(), 5));
        }
        Ok(Self(
            h.nodes[0], h.nodes[1], h.nodes[2], h.nodes[3], h.nodes[4],
        ))
    }
}

impl Index<usize> for Hx {
    type Output = NodeId;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

impl Index<usize> for WHx {
    type Output = NodeId;

    fn index(&self, index: usize) -> &Self::Output {
        &self.edge[index]
    }
}

// -- iter
impl<'a> IntoIterator for &'a H2 {
    type Item = &'a NodeId;
    type IntoIter = std::array::IntoIter<&'a NodeId, 2>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::IntoIterator::into_iter([&self.0, &self.1])
    }
}

impl<'a> IntoIterator for &'a H3 {
    type Item = &'a NodeId;
    type IntoIter = std::array::IntoIter<&'a NodeId, 3>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::IntoIterator::into_iter([&self.0, &self.1, &self.2])
    }
}

impl<'a> IntoIterator for &'a H4 {
    type Item = &'a NodeId;
    type IntoIter = std::array::IntoIter<&'a NodeId, 4>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::IntoIterator::into_iter([&self.0, &self.1, &self.2, &self.3])
    }
}

impl<'a> IntoIterator for &'a H5 {
    type Item = &'a NodeId;
    type IntoIter = std::array::IntoIter<&'a NodeId, 5>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::IntoIterator::into_iter([&self.0, &self.1, &self.2, &self.3, &self.4])
    }
}

impl<'a> IntoIterator for &'a Hx {
    type Item = &'a NodeId;
    type IntoIter = std::slice::Iter<'a, NodeId>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.iter()
    }
}

impl<'a> IntoIterator for &'a WHx {
    type Item = &'a NodeId;
    type IntoIter = std::slice::Iter<'a, NodeId>;

    fn into_iter(self) -> Self::IntoIter {
        self.edge.nodes.iter()
    }
}

// -- into iter
impl IntoIterator for H2 {
    type Item = NodeId;
    type IntoIter = std::array::IntoIter<NodeId, 2>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::IntoIterator::into_iter([self.0, self.1])
    }
}

impl IntoIterator for H3 {
    type Item = NodeId;
    type IntoIter = std::array::IntoIter<NodeId, 3>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::IntoIterator::into_iter([self.0, self.1, self.2])
    }
}

impl IntoIterator for H4 {
    type Item = NodeId;
    type IntoIter = std::array::IntoIter<NodeId, 4>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::IntoIterator::into_iter([self.0, self.1, self.2, self.3])
    }
}

impl IntoIterator for H5 {
    type Item = NodeId;
    type IntoIter = std::array::IntoIter<NodeId, 5>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::IntoIterator::into_iter([self.0, self.1, self.2, self.3, self.4])
    }
}

impl IntoIterator for Hx {
    type Item = NodeId;
    type IntoIter = std::vec::IntoIter<NodeId>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.into_iter()
    }
}

impl IntoIterator for WHx {
    type Item = NodeId;
    type IntoIter = std::vec::IntoIter<NodeId>;

    fn into_iter(self) -> Self::IntoIter {
        self.edge.into_iter()
    }
}

// -- iter mut
impl<'a> IntoIterator for &'a mut H2 {
    type Item = &'a mut NodeId;
    type IntoIter = std::array::IntoIter<&'a mut NodeId, 2>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::IntoIterator::into_iter([&mut self.0, &mut self.1])
    }
}

impl<'a> IntoIterator for &'a mut H3 {
    type Item = &'a mut NodeId;
    type IntoIter = std::array::IntoIter<&'a mut NodeId, 3>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::IntoIterator::into_iter([&mut self.0, &mut self.1, &mut self.2])
    }
}

impl<'a> IntoIterator for &'a mut H4 {
    type Item = &'a mut NodeId;
    type IntoIter = std::array::IntoIter<&'a mut NodeId, 4>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::IntoIterator::into_iter([&mut self.0, &mut self.1, &mut self.2, &mut self.3])
    }
}

impl<'a> IntoIterator for &'a mut H5 {
    type Item = &'a mut NodeId;
    type IntoIter = std::array::IntoIter<&'a mut NodeId, 5>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::IntoIterator::into_iter([
            &mut self.0,
            &mut self.1,
            &mut self.2,
            &mut self.3,
            &mut self.4,
        ])
    }
}

impl<'a> IntoIterator for &'a mut Hx {
    type Item = &'a mut NodeId;
    type IntoIter = std::slice::IterMut<'a, NodeId>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.iter_mut()
    }
}

impl<'a> IntoIterator for &'a mut WHx {
    type Item = &'a mut NodeId;
    type IntoIter = std::slice::IterMut<'a, NodeId>;

    fn into_iter(self) -> Self::IntoIter {
        self.edge.nodes.iter_mut()
    }
}

#[derive(FromPyObject, Debug)]
pub enum Hypergraph<'py> {
    #[pyo3(transparent)]
    Db(Bound<'py, UnweightedHypergraph>),
    #[pyo3(transparent)]
    File(Bound<'py, ArchivedUnweightedHypergraph>),
}

impl_stub_type!(Hypergraph<'_> = UnweightedHypergraph | ArchivedUnweightedHypergraph);

submit! {
    gen_methods_from_python! {
        r#"
        class H2:
            def __init__(u: int, v:int) -> H2: ...
        "#
    };
    gen_methods_from_python! {
        r#"
        class H3:
            def __init__(u: int, v:int, w:int) -> H3: ...
        "#
    };
    gen_methods_from_python! {
        r#"
        class H4:
            def __init__(u: int, v:int, w:int) -> H4: ...
        "#
    };
    gen_methods_from_python! {
        r#"
        class H5:
            def __init__(u: int, v:int, w:int) -> H5: ...
        "#
    };

    gen_methods_from_python! {
        r#"
        class WH2:
            def __init__(u: int, v:int) -> WH2: ...
        "#
    };
    gen_methods_from_python! {
        r#"
        class WH3:
            def __init__(u: int, v:int, w:int) -> WH3: ...
        "#
    };
    gen_methods_from_python! {
        r#"
        class WH4:
            def __init__(u: int, v:int, w:int) -> WH4: ...
        "#
    };
    gen_methods_from_python! {
        r#"
        class WH5:
            def __init__(u: int, v:int, w:int) -> WH5: ...
        "#
    }
}
