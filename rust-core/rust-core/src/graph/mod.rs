pub mod adj_list;
pub mod edge_collection;
pub mod error;
pub mod flat_adj_list;
pub mod hyper_adj_list;
pub mod hypergraph;
pub mod serialize;
pub mod types;

pub use adj_list::*;
pub use edge_collection::*;
pub use error::*;
pub use flat_adj_list::*;
pub use hypergraph::*;
pub use types::*;

use pyo3::pymodule;
use pyo3_stub_gen::reexport_module_members;

#[pymodule(submodule)]
pub mod graph {

    #[pymodule_export]
    use super::adj_list::AdjList;

    #[pymodule_export]
    use super::hypergraph::UnweightedHypergraph;

    #[pymodule_export]
    use super::hypergraph::WeightedHypergraph;

    #[pymodule_export]
    use crate::loader::DatasetLoader;
}

reexport_module_members!("rust_core.graph" from "rust_core._core.graph");
