pub mod common;
pub mod conference;
pub mod dblp;
pub mod facebook_hs;
pub mod friendship_hs;
pub mod gene_disease;
pub mod geology;
pub mod high_school;
pub mod history;
pub mod hospital;
pub mod justice;
pub mod pacs;
pub mod primary_school;
pub mod wiki_talk;
pub mod workspace;
// pub mod babbuini;
pub mod enron;
pub mod eu;
pub mod ndc_classes;
pub mod ndc_substances;
pub mod wiki;

use crate::loader::common::Loader;
use pyo3::{exceptions::PyIOError, prelude::*};
use pyo3_stub_gen::reexport_module_members;
use std::path::PathBuf;

fn map_rv<T, U>(
    res: Result<T, Box<dyn std::error::Error>>,
    dataset_location: PathBuf,
) -> PyResult<U>
where
    T: Into<U>,
{
    res.map_err(|e| {
        PyIOError::new_err(format!(
            "Failed to read '{}': {}",
            dataset_location.display(),
            e
        ))
    })
    .map(|hg| hg.into())
}

// This macro will generate the entire module at once
macro_rules! generate_loader_module {
    ($(($name:ident, $rv:ty, $loader_path:path)),* $(,)?) => {
        #[pymodule(submodule)]
        pub mod loader {
            use std::path::PathBuf;
            use pyo3::{exceptions::PyIOError, prelude::*};
            use pyo3_stub_gen::derive::gen_stub_pyfunction;

            use crate::{
                graph::{UnweightedHypergraph, WeightedHypergraph, AdjList},
                loader::common::Loader,
            };

            #[pyfunction]
            #[gen_stub_pyfunction(module = "rust_core.core.loader")]
            #[pyo3(signature = (dataset_location, cache_dir = None))]
            pub fn test(dataset_location: std::path::PathBuf, cache_dir: Option<std::path::PathBuf>) {
                println!("Test");
            }

            $(
                #[doc = concat!("Loads a dataset into a `", stringify!($rv), "` wrapper.")]
                #[doc = ""]
                #[doc = "This function will attempt to read from the raw `dataset_location` file."]
                #[doc = "If a `cache_dir` is provided, it will prioritize loading from or saving to the cache via `.load()`,"]
                #[doc = "otherwise it falls back to parsing the raw file via `.from_file()`."]
                #[doc = ""]
                #[doc = concat!("Underlying loader type: `", stringify!($loader_path), "`")]
                #[pyfunction]
                #[gen_stub_pyfunction(module = "rust_core.core.loader")]
                #[pyo3(signature = (dataset_location, cache_dir = None))]
                pub fn $name(
                    dataset_location: std::path::PathBuf,
                    cache_dir: Option<std::path::PathBuf>,
                ) -> pyo3::PyResult<$rv> {
                    super::map_rv(
                        match cache_dir {
                            Some(cache_dir) => <$loader_path>::load(&dataset_location, &cache_dir),
                            None => <$loader_path>::from_file(&dataset_location),
                        },
                        dataset_location,
                    )
                }
            )*
        }
    };
}

// Invoke the macro to build the entire loader module
generate_loader_module!(
    (
        load_conference_uw,
        UnweightedHypergraph,
        super::conference::Unweighted
    ),
    (
        load_conference_w,
        WeightedHypergraph,
        super::conference::Weighted
    ),
    (
        load_primary_school_uw,
        UnweightedHypergraph,
        super::primary_school::Unweighted
    ),
    (
        load_primary_school_w,
        WeightedHypergraph,
        super::primary_school::Weighted
    ),
    (
        load_high_school_uw,
        UnweightedHypergraph,
        super::high_school::Unweighted
    ),
    (
        load_high_school_w,
        WeightedHypergraph,
        super::high_school::Weighted
    ),
    (
        load_hospital_uw,
        UnweightedHypergraph,
        super::hospital::Unweighted
    ),
    (
        load_hospital_w,
        WeightedHypergraph,
        super::hospital::Weighted
    ),
    (
        load_facebook_hs,
        UnweightedHypergraph,
        super::facebook_hs::Unweighted
    ),
    (
        load_friendship_hs_uw,
        UnweightedHypergraph,
        super::friendship_hs::Unweighted
    ),
    (
        load_friendship_hs_w,
        WeightedHypergraph,
        super::friendship_hs::Weighted
    ),
    (
        load_gene_disease,
        WeightedHypergraph,
        super::gene_disease::Weighted
    ),
    (load_pacs_uw, UnweightedHypergraph, super::pacs::Unweighted),
    (load_pacs_w, WeightedHypergraph, super::pacs::Weighted),
    (
        load_workspace_uw,
        UnweightedHypergraph,
        super::workspace::Unweighted
    ),
    (
        load_workspace_w,
        WeightedHypergraph,
        super::workspace::Weighted
    ),
    (load_dblp_uw, UnweightedHypergraph, super::dblp::Unweighted),
    (load_dblp_w, WeightedHypergraph, super::dblp::Weighted),
    (
        load_history_uw,
        UnweightedHypergraph,
        super::history::Unweighted
    ),
    (load_history_w, WeightedHypergraph, super::history::Weighted),
    (
        load_geology_uw,
        UnweightedHypergraph,
        super::geology::Unweighted
    ),
    (load_geology_w, WeightedHypergraph, super::geology::Weighted),
    (
        load_justice_uw,
        UnweightedHypergraph,
        super::justice::Unweighted
    ),
    (load_justice_w, WeightedHypergraph, super::justice::Weighted),
    // (load_babbuini_uw, UnweightedHypergraph, super::babbuini::Unweighted),
    // (load_babbuini_w, WeightedHypergraph, super::babbuini::Weighted),
    (load_wiki_uw, UnweightedHypergraph, super::wiki::Unweighted),
    (load_wiki_w, WeightedHypergraph, super::wiki::Weighted),
    (
        load_ndc_substances_uw,
        UnweightedHypergraph,
        super::ndc_substances::Unweighted
    ),
    (
        load_ndc_substances_w,
        WeightedHypergraph,
        super::ndc_substances::Weighted
    ),
    (
        load_ndc_classes_uw,
        UnweightedHypergraph,
        super::ndc_classes::Unweighted
    ),
    (
        load_ndc_classes_w,
        WeightedHypergraph,
        super::ndc_classes::Weighted
    ),
    (load_eu_uw, UnweightedHypergraph, super::eu::Unweighted),
    (load_eu_w, WeightedHypergraph, super::eu::Weighted),
    (
        load_enron_uw,
        UnweightedHypergraph,
        super::enron::Unweighted
    ),
    (load_enron_w, WeightedHypergraph, super::enron::Weighted),
    (
        load_wiki_talk_2_uniform,
        AdjList,
        super::wiki_talk::Unweighted2Uniform
    )
);

reexport_module_members!("rust_core.loader" from "rust_core.core.loader");
