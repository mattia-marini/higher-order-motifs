use pyo3::prelude::*;
use pyo3_stub_gen::reexport_module_members;
pub mod algorithms;
pub mod compressed_motif;
pub mod compressed_node_set;
pub mod fingerprint;
pub mod types;

#[pymodule(submodule)]
pub mod motifs {
    use crate::graph::{PyHypergraph, UnweightedHypergraph, WeightedHypergraph};
    use pyo3::pyfunction;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    /// Computed
    pub fn analyze_esu_based_3(hg: PyHypergraph) {
        match hg {
            PyHypergraph::Unweighted(unweighted) => {
                super::algorithms::esu_based::unweighted_3(&unweighted);
            }
            PyHypergraph::Weighted(weighted) => {
                super::algorithms::esu_based::weighted_3(&weighted);
            }
        }
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    pub fn analyze_esu_based_4(hg: PyHypergraph) {
        match hg {
            PyHypergraph::Unweighted(unweighted) => {
                super::algorithms::esu_based::unweighted_4(&unweighted);
            }
            PyHypergraph::Weighted(weighted) => {
                super::algorithms::esu_based::weighted_4(&weighted);
            }
        }
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    pub fn orca_3(hg: PyHypergraph) {
        match hg {
            PyHypergraph::Unweighted(unweighted) => {
                super::algorithms::orca::unweighted_3(&unweighted);
            }
            PyHypergraph::Weighted(weighted) => {
                super::algorithms::orca::weighted_3(&weighted);
            }
        }
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    pub fn orca_4(hg: PyHypergraph) {
        match hg {
            PyHypergraph::Unweighted(unweighted) => {
                super::algorithms::orca::unweighted_4(&unweighted);
            }
            PyHypergraph::Weighted(weighted) => {
                super::algorithms::orca::weighted_4(&weighted);
            }
        }
    }
}

reexport_module_members!("rust_core.motifs" from "rust_core._core.motifs");
