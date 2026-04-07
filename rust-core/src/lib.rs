pub mod motifs;
pub mod triangle;
use pyo3::prelude::*;

use pyo3_stub_gen::{define_stub_info_gatherer, reexport_module_members};

pub mod util;

#[pymodule]
pub mod core {

    use crate::util::submodules_initializer::PyModuleSubmoduleExt;
    use pyo3::{Bound, PyResult, types::PyModule};

    #[pymodule_export]
    use super::triangle::triangle;

    #[pymodule_export]
    use super::motifs::motifs;

    #[pymodule_init]
    pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.init_submodules()?;
        Ok(())
    }
}

reexport_module_members!("rust_core" from "rust_core.core");

define_stub_info_gatherer!(stub_info);
