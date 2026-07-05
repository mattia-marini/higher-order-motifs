pub mod bindings;
pub mod error;
pub mod graph;
pub mod hypergraph;

pub use graph::*;
pub use hypergraph::*;

pub type NodeId = u32;
pub type EdgeId = u32;
pub type NodeWeight = f32;
