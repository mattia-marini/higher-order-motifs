pub mod adj_list;
pub mod flat_adj_list;

pub mod error;
pub mod traits;
pub mod types;
pub mod unweighted_hypergraph;
pub mod weighted_hypergraph;

pub use adj_list::*;
pub use flat_adj_list::*;
pub use types::*;
pub use unweighted_hypergraph::*;
pub use weighted_hypergraph::*;

use pyo3::pymodule;
use pyo3_stub_gen::reexport_module_members;

#[pymodule(submodule)]
pub mod graph {

    #[pymodule_export]
    use super::unweighted_hypergraph::UnweightedHypergraph;

    #[pymodule_export]
    use super::types::H2;

    #[pymodule_export]
    use super::types::H3;

    #[pymodule_export]
    use super::types::H4;

    #[pymodule_export]
    use super::types::H5;

    #[pymodule_export]
    use super::types::WH2;

    #[pymodule_export]
    use super::types::WH3;

    #[pymodule_export]
    use super::types::WH4;

    #[pymodule_export]
    use super::types::WH5;
}

reexport_module_members!("rust_core.graph" from "rust_core.core.graph");
