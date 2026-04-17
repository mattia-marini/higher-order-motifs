use pyo3::prelude::*;
use pyo3_stub_gen::reexport_module_members;
pub mod motifs3;
pub mod orca;

#[pymodule(submodule)]
pub mod motifs {
    #[pymodule_export]
    use super::motifs3::count_motifs_3;

    #[pymodule_export]
    use super::orca::orca;
}

reexport_module_members!("rust_core.motifs" from "rust_core.core.motifs");
