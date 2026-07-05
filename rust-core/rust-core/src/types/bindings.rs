#[cfg(feature = "bindings")]
#[pyo3::pymodule(submodule)]
pub mod graph {
    use pyo3::pymodule;
    use pyo3_stub_gen::reexport_module_members;

    #[pymodule_export]
    use super::super::adj_list::UnweightedUndirectedAdjList;

    #[pymodule_export]
    use super::super::adj_list::WeightedUndirectedAdjList;

    #[pymodule_export]
    use super::super::adj_list::UnweightedDirectedAdjList;

    #[pymodule_export]
    use super::super::adj_list::WeightedDirectedAdjList;

    #[pymodule_export]
    use super::super::adj_list::UnweightedUndirectedAdjSet;

    #[pymodule_export]
    use super::super::adj_list::WeightedUndirectedAdjSet;

    #[pymodule_export]
    use super::super::adj_list::UnweightedDirectedAdjSet;

    #[pymodule_export]
    use super::super::adj_list::WeightedDirectedAdjSet;

    reexport_module_members!("rust_core.graph" from "rust_core._core.graph");
}

#[cfg(feature = "bindings")]
#[pyo3::pymodule(submodule)]
pub mod hypergraph {
    use pyo3::pymodule;
    use pyo3_stub_gen::reexport_module_members;

    #[pymodule_export]
    use super::super::hypergraph::UnweightedHypergraph;

    #[pymodule_export]
    use super::super::hypergraph::WeightedHypergraph;

    reexport_module_members!("rust_core.hypergraph" from "rust_core._core.hypergraph");
}
