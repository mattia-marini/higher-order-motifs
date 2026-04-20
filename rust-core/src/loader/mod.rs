pub mod common;
pub mod med;
pub mod small;
pub mod wiki_talk;

pub use med::*;
pub use small::*;
pub use wiki_talk::*;

use pyo3::prelude::*;

#[pymodule]
pub mod loader {
    use std::path::PathBuf;

    use pyo3::{exceptions::PyIOError, prelude::*};
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    use crate::graph::Hypergraph;

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.loader")]
    pub fn load_wiki_talk(
        dataset_dir: PathBuf,
        cache_dir: Option<PathBuf>,
    ) -> PyResult<Hypergraph> {
        super::load_wiki_talk_cached(&dataset_dir, cache_dir.as_ref()).map_err(|e| {
            PyIOError::new_err(format!("Failed to read {:?} dataset: {}", dataset_dir, e))
        })
    }
}
