use pyo3::prelude::*;
use pyo3_stub_gen::reexport_module_members;
pub mod base;
pub mod motifs3;
pub mod orca;

#[pymodule(submodule)]
pub mod motifs {
    use crate::graph::WeightedHypergraph;
    use pyo3::pyfunction;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.motifs")]
    pub fn count_motifs_3(hg: &WeightedHypergraph) {
        super::motifs3::count_motifs_3(hg)
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.motifs")]
    pub fn count_motifs_4(hg: &WeightedHypergraph) {
        super::motifs3::count_motifs_4(hg);
    }

    #[pymodule_export]
    use super::orca::orca;
}

reexport_module_members!("rust_core.motifs" from "rust_core.core.motifs");
