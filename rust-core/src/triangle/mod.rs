use pyo3::prelude::*;

mod cetc;
mod common;
mod forward;
mod kclist;

#[pymodule(submodule)]
pub mod triangle {

    #[pymodule_export]
    use super::forward::forward;

    #[pymodule_export]
    use super::cetc::cetc;

    #[pymodule_export]
    use super::common::common;

    #[pymodule_export]
    use super::kclist::kclist;
}
