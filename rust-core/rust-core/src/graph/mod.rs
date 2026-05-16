// pub mod adj_list;
// pub mod flat_adj_list;
// pub mod test;
// pub mod traits2;
pub mod edge_collection;
pub mod error;
pub mod types2;
pub mod unweighted_hypergraph2;

// pub mod traits;
// pub mod types;
// pub mod unweighted_hypergraph;
// pub mod weighted_hypergraph;

// pub use adj_list::*;
// pub use flat_adj_list::*;
// pub use types::*;
// pub use unweighted_hypergraph::*;
// pub use weighted_hypergraph::*;

use pyo3::pymodule;
use pyo3_stub_gen::reexport_module_members;

#[pymodule(submodule)]
pub mod graph {

    #[pymodule_export]
    use super::unweighted_hypergraph2::UnweightedHypergraph;

    #[pymodule_export]
    use super::unweighted_hypergraph2::WeightedHypergraph;
}

reexport_module_members!("rust_core.graph" from "rust_core.core.graph");
