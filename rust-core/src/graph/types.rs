pub type NodeId = u32;
pub type NodeWeight = f32;

pub type H2 = (NodeId, NodeId);
pub type H3 = (NodeId, NodeId, NodeId);
pub type H4 = (NodeId, NodeId, NodeId, NodeId);
pub type H5 = (NodeId, NodeId, NodeId, NodeId, NodeId);

pub type WH2 = (NodeWeight, (NodeId, NodeId));
pub type WH3 = (NodeWeight, (NodeId, NodeId, NodeId));
pub type WH4 = (NodeWeight, (NodeId, NodeId, NodeId, NodeId));
pub type WH5 = (NodeWeight, (NodeId, NodeId, NodeId, NodeId, NodeId));

// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, FromPyObject)]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// #[pyclass]
// pub struct H2(pub NodeId, pub NodeId);
//
// // #[pymethods]
// // #[gen_stub_pymethods(module = "rust_core.core.graph")]
// // impl H2 {
// //     #[new]
// //     pub fn new(u: NodeId, v: NodeId) -> Self {
// //         H2(u, v)
// //     }
// // }
//
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, FromPyObject)]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// #[pyclass]
// pub struct H3(pub NodeId, pub NodeId, pub NodeId);
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl H3 {
//     #[new]
//     pub fn new(u: NodeId, v: NodeId, w: NodeId) -> Self {
//         H3(u, v, w)
//     }
// }
//
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, FromPyObject)]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// #[pyclass]
// pub struct H4(pub NodeId, pub NodeId, pub NodeId, pub NodeId);
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl H4 {
//     #[new]
//     pub fn new(u: NodeId, v: NodeId, w: NodeId, x: NodeId) -> Self {
//         H4(u, v, w, x)
//     }
// }
//
// #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, FromPyObject)]
// #[gen_stub_pyclass(module = "rust_core.core.graph")]
// #[pyclass]
// pub struct H5(pub NodeId, pub NodeId, pub NodeId, pub NodeId, pub NodeId);
//
// #[pymethods]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl H5 {
//     #[new]
//     pub fn new(u: NodeId, v: NodeId, w: NodeId, x: NodeId, y: NodeId) -> Self {
//         H5(u, v, w, x, y)
//     }
// }
