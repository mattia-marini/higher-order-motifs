use pyo3::prelude::*;
use pyo3_stub_gen::reexport_module_members;
pub mod algorithms;
pub mod compressed_motif;
pub mod compressed_node_set;
pub mod fingerprint;

#[pymodule(submodule)]
pub mod motifs {
    use crate::graph::{PyHypergraph, UnweightedHypergraph, WeightedHypergraph};
    use pyo3::pyfunction;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.motifs")]
    pub fn count_motifs_3_unweighted(hg: &UnweightedHypergraph) {
        super::motifs3::count_motifs_3_unweighted(hg)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.motifs")]
    pub fn count_motifs_3_weighted(hg: &WeightedHypergraph) {
        super::motifs3::count_motifs_3_weighted(hg)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.motifs")]
    pub fn count_motifs_4(hg: PyHypergraph) {
        match hg {
            PyHypergraph::Unweighted(unweighted) => {
                super::motifs3::count_motifs_3_unweighted(&unweighted);
            }
            PyHypergraph::Weighted(weighted) => {
                super::motifs3::count_motifs_3_weighted(&weighted);
            }
        }
    }

    #[pymodule_export]
    use super::orca::orca;
}

reexport_module_members!("rust_core.motifs" from "rust_core.core.motifs");
