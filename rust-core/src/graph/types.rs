use pyo3::pyclass;

pub type NodeId = u32;

#[pyclass]
pub struct H2(NodeId, NodeId);

impl H2 {
    pub fn new(u: NodeId, v: NodeId) -> Self {
        H2(u, v)
    }
}

#[pyclass]
pub struct H3(NodeId, NodeId, NodeId);

impl H3 {
    pub fn new(u: NodeId, v: NodeId, w: NodeId) -> Self {
        H3(u, v, w)
    }
}

#[pyclass]
pub struct H4(NodeId, NodeId, NodeId, NodeId);

impl H4 {
    pub fn new(u: NodeId, v: NodeId, w: NodeId, x: NodeId) -> Self {
        H4(u, v, w, x)
    }
}

#[pyclass]
pub struct H5(NodeId, NodeId, NodeId, NodeId, NodeId);

impl H5 {
    pub fn new(u: NodeId, v: NodeId, w: NodeId, x: NodeId, y: NodeId) -> Self {
        H5(u, v, w, x, y)
    }
}