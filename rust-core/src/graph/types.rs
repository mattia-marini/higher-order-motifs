use pyo3::{FromPyObject, pyclass, pymethods};
use pyo3_stub_gen::derive::gen_stub_pyclass;
use rkyv::{Archive, Deserialize, Serialize};

pub type NodeId = u32;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, FromPyObject)]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
#[pyclass]
pub struct H2(NodeId, NodeId);

#[pymethods]
impl H2 {
    #[new]
    pub fn new(u: NodeId, v: NodeId) -> Self {
        H2(u, v)
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, FromPyObject)]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
#[pyclass]
pub struct H3(NodeId, NodeId, NodeId);

#[pymethods]
impl H3 {
    #[new]
    pub fn new(u: NodeId, v: NodeId, w: NodeId) -> Self {
        H3(u, v, w)
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, FromPyObject)]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
#[pyclass]
pub struct H4(NodeId, NodeId, NodeId, NodeId);

#[pymethods]
impl H4 {
    #[new]
    pub fn new(u: NodeId, v: NodeId, w: NodeId, x: NodeId) -> Self {
        H4(u, v, w, x)
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, FromPyObject)]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
#[pyclass]
pub struct H5(NodeId, NodeId, NodeId, NodeId, NodeId);

#[pymethods]
impl H5 {
    #[new]
    pub fn new(u: NodeId, v: NodeId, w: NodeId, x: NodeId, y: NodeId) -> Self {
        H5(u, v, w, x, y)
    }
}
