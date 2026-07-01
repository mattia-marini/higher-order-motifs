#![allow(unused)]
pub mod loader;
pub mod misc;
pub mod motifs;
pub mod triangle;
pub mod types;
pub mod util;

#[cfg(feature = "bindings")]
#[cfg_attr(feature = "bindings", pyo3::pymodule)]
pub mod _core {
    use crate::util::submodules_initializer::PyModuleSubmoduleExt;
    use pyo3::prelude::*;
    use pyo3_stub_gen::{define_stub_info_gatherer, exclude_from_all, reexport_module_members};

    #[pymodule_export]
    use super::triangle::triangle;

    #[pymodule_export]
    use super::motifs::motifs;

    #[pymodule_export]
    use super::graph::graph;

    #[pymodule_export]
    use super::loader::loader;

    pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.init_submodules()?;
        Ok(())
    }
    reexport_module_members!("rust_core" from "rust_core._core");
    exclude_from_all!("rust_core", "loader");
    exclude_from_all!("rust_core._core", "loader");
}

#[cfg(feature = "bindings")]
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let project_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "CARGO_MANIFEST_DIR has no parent",
        ))?;

    let infos = pyo3_stub_gen::StubInfo::from_pyproject_toml(workspace_dir.join("pyproject.toml"))?;
    Ok(infos)
}
