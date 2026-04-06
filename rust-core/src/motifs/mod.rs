use pyo3::prelude::*;
pub mod motifs3;

#[pymodule(submodule)]
pub mod motifs {
    #[pymodule_export]
    use super::motifs3::count_motifs_3;
}
