#![allow(unused)]
pub mod graph;
pub mod loader;
pub mod misc;
pub mod motifs;
// pub mod triangle;

#[cfg(test)]
mod test;

use pyo3::prelude::*;

use pyo3_stub_gen::{StubInfo, define_stub_info_gatherer, reexport_module_members};

pub mod util;

#[pymodule]
pub mod core {
    use crate::util::submodules_initializer::PyModuleSubmoduleExt;
    use pyo3::{Bound, PyResult, types::PyModule};

    // use crate::util::submodules_initializer::PyModuleSubmoduleExt;
    // use pyo3::{Bound, PyResult, types::PyModule};
    //
    // #[pymodule_export]
    // use super::triangle::triangle;
    //
    #[pymodule_export]
    use super::motifs::motifs;

    #[pymodule_export]
    use super::graph::graph;

    #[pymodule_export]
    use super::loader::loader;

    #[pymodule_init]
    pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.init_submodules()?;
        Ok(())
    }
}

reexport_module_members!("rust_core" from "rust_core.core");

pub fn stub_info() -> pyo3_stub_gen::Result<StubInfo> {
    let project_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "CARGO_MANIFEST_DIR has no parent",
        ))?;

    let infos = StubInfo::from_pyproject_toml(workspace_dir.join("pyproject.toml"))?;
    Ok(infos)
}
