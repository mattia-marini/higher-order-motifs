pub mod common;
pub mod conference;
// pub mod wiki_talk;

use pyo3::prelude::*;

pub use conference::load_conference;
// pub use wiki_talk::load_wiki_talk;

#[pymodule]
pub mod loader {
    use std::path::PathBuf;

    use pyo3::{exceptions::PyIOError, prelude::*};
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    use crate::graph::UnweightedHypergraph;

    // #[pyfunction]
    // #[gen_stub_pyfunction(module = "rust_core.core.loader")]
    // #[pyo3(signature = (dataset_dir, cache_dir = None))]
    // pub fn load_wiki_talk(
    //     dataset_dir: PathBuf,
    //     cache_dir: Option<PathBuf>,
    // ) -> PyResult<UnweightedHypergraph> {
    //     super::load_wiki_talk(&dataset_dir, cache_dir.as_ref()).map_err(|e| {
    //         PyIOError::new_err(format!("Failed to read {:?} dataset: {}", dataset_dir, e))
    //     })
    // }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.loader")]
    // #[pyo3(signature = (dataset_dir, cache_dir))]
    pub fn load_conference(
        dataset_dir: PathBuf,
        cache_dir: PathBuf,
    ) -> PyResult<UnweightedHypergraph> {
        super::load_conference(&dataset_dir, &cache_dir)
            .map_err(|e| {
                PyIOError::new_err(format!("Failed to read {:?} dataset: {}", dataset_dir, e))
            })
            .map(|hg| UnweightedHypergraph(hg))
    }
}
