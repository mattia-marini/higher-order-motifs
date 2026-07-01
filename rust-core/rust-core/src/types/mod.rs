pub mod error;
pub mod graph;
pub mod hypergraph;

pub use graph::*;
pub use hypergraph::*;

pub type NodeId = u32;
pub type EdgeId = u32;
pub type NodeWeight = f32;

#[cfg(feature = "bindings")]
#[pyo3::pymodule(submodule)]
pub mod types {
    use pyo3::pymodule;
    use pyo3_stub_gen::reexport_module_members;

    #[pymodule_export]
    use super::graph::incidence::UnweightedIncList;

    #[pymodule_export]
    use super::graph::incidence::WeightedIncList;

    #[pymodule_export]
    use super::graph::adjacency::UnweightedAdjList;

    #[pymodule_export]
    use super::graph::adjacency::WeightedAdjList;

    #[pymodule_export]
    use super::hypergraph::hypergraph::UnweightedHypergraph;

    #[pymodule_export]
    use super::hypergraph::hypergraph::WeightedHypergraph;

    reexport_module_members!("rust_core.graph" from "rust_core._core.graph");
}
