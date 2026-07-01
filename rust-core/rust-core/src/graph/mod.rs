pub mod adj_list;
pub mod edge_collection;
pub mod error;
pub mod hyperedge;
pub mod hypergraph;
pub mod inc_list;
pub mod serialize;

pub use adj_list::*;
pub use edge_collection::*;
pub use error::*;
pub use hyperedge::*;
pub use hypergraph::*;

#[cfg(feature = "bindings")]
#[cfg_attr(feature = "bindings", pyo3::pymodule(submodule))]
pub mod graph {
    use pyo3::pymodule;
    use pyo3_stub_gen::reexport_module_members;

    #[pymodule_export]
    use super::adj_list::UnweightedAdjList;

    #[pymodule_export]
    use super::adj_list::WeightedAdjList;

    #[pymodule_export]
    use super::hypergraph::UnweightedHypergraph;

    #[pymodule_export]
    use super::hypergraph::WeightedHypergraph;

    #[pymodule_export]
    use crate::loader::DatasetLoader;

    reexport_module_members!("rust_core.graph" from "rust_core._core.graph");
}
