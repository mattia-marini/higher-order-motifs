use pyo3::prelude::*;
use pyo3_stub_gen::reexport_module_members;

pub mod cbs;
pub mod cetc;
pub mod forward;
pub mod kclist;

#[pymodule(submodule)]
pub mod triangle {
    #[pymodule_export]
    use super::forward::forward;

    #[pymodule_export]
    use super::cetc::cetc;

    #[pymodule_export]
    use super::kclist::kclist;
}

reexport_module_members!("rust_core.triangle" from "rust_core.core.triangle");
