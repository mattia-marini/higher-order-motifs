pub mod common;
pub mod conference;
pub mod dblp;
pub mod descriptors;
pub mod enron;
pub mod eu;
pub mod facebook_hs;
pub mod friendship_hs;
pub mod gene_disease;
pub mod geology;
pub mod high_school;
pub mod history;
pub mod hospital;
pub mod justice;
pub mod ndc_classes;
pub mod ndc_substances;
pub mod pacs;
pub mod primary_school;
pub mod wiki;
pub mod wiki_talk;
pub mod workspace;
// pub mod babbuini;

//
// use crate::loader::common::Loader;
// use pyo3::{exceptions::PyIOError, prelude::*};
// use std::path::PathBuf;
//
// fn map_rv<T, U>(
//     res: Result<T, Box<dyn std::error::Error>>,
//     dataset_location: PathBuf,
// ) -> PyResult<U>
// where
//     T: Into<U>,
// {
//     res.map_err(|e| {
//         PyIOError::new_err(format!(
//             "Failed to read '{}': {}",
//             dataset_location.display(),
//             e
//         ))
//     })
//     .map(|hg| hg.into())
// }
//
// // This macro will generate the entire module at once
// macro_rules! generate_loader_module {
//     ($(($name:ident, $rv:ty, $loader_path:path)),* $(,)?) => {
//         #[pymodule(submodule)]
//         pub mod loader {
//             use std::path::PathBuf;
//             use pyo3::{exceptions::PyIOError, prelude::*};
//             use pyo3_stub_gen::derive::gen_stub_pyfunction;
//
//             use crate::{
//                 graph::{UnweightedHypergraph, WeightedHypergraph, AdjList},
//                 loader::common::Loader,
//             };
//
//             #[pyfunction]
//             #[gen_stub_pyfunction(module = "rust_core.core.loader")]
//             #[pyo3(signature = (dataset_location, cache_dir = None))]
//             pub fn test(dataset_location: std::path::PathBuf, cache_dir: Option<std::path::PathBuf>) {
//                 println!("Test");
//             }
//
//             $(
//                 #[doc = concat!("Loads a dataset into a `", stringify!($rv), "` wrapper.")]
//                 #[doc = ""]
//                 #[doc = "This function will attempt to read from the raw `dataset_location` file."]
//                 #[doc = "If a `cache_dir` is provided, it will prioritize loading from or saving to the cache via `.load()`,"]
//                 #[doc = "otherwise it falls back to parsing the raw file via `.from_file()`."]
//                 #[doc = ""]
//                 #[doc = concat!("Underlying loader type: `", stringify!($loader_path), "`")]
//                 #[pyfunction]
//                 #[gen_stub_pyfunction(module = "rust_core.core.loader")]
//                 #[pyo3(signature = (dataset_location, cache_dir = None))]
//                 pub fn $name(
//                     dataset_location: std::path::PathBuf,
//                     cache_dir: Option<std::path::PathBuf>,
//                 ) -> pyo3::PyResult<$rv> {
//                     super::map_rv(
//                         match cache_dir {
//                             Some(cache_dir) => <$loader_path>::load(&dataset_location, &cache_dir),
//                             None => <$loader_path>::from_file(&dataset_location),
//                         },
//                         dataset_location,
//                     )
//                 }
//             )*
//         }
//     };
// }
//
// // Invoke the macro to build the entire loader module
// #[rustfmt::skip]
// generate_loader_module!(
//     (load_conference_uw, UnweightedHypergraph, super::descriptors::ConferenceStdUnweightedLoader),
//     (load_conference_w, WeightedHypergraph, super::descriptors::ConferenceStdWeightedLoader),
//     (load_primary_school_uw, UnweightedHypergraph, super::descriptors::PrimarySchoolStdUnweightedLoader),
//     (load_primary_school_w, WeightedHypergraph, super::descriptors::PrimarySchoolStdWeightedLoader),
//     (load_high_school_uw, UnweightedHypergraph, super::descriptors::HighSchoolStdUnweightedLoader),
//     (load_high_school_w, WeightedHypergraph, super::descriptors::HighSchoolStdWeightedLoader),
//     (load_hospital_uw, UnweightedHypergraph, super::descriptors::HospitalStdUnweightedLoader),
//     (load_hospital_w, WeightedHypergraph, super::descriptors::HospitalStdWeightedLoader),
//     (load_facebook_hs, UnweightedHypergraph, super::descriptors::FacebookHsStdUnweightedLoader),
//     (load_friendship_hs_uw, UnweightedHypergraph, super::descriptors::FriendshipHsStdUnweightedLoader),
//     (load_friendship_hs_w, WeightedHypergraph, super::descriptors::FriendshipHsStdWeightedLoader),
//     (load_gene_disease, WeightedHypergraph, super::descriptors::GeneDiseaseStdWeightedLoader),
//     (load_pacs_uw, UnweightedHypergraph, super::descriptors::PacsStdUnweightedLoader),
//     (load_pacs_w, WeightedHypergraph, super::descriptors::PacsStdWeightedLoader),
//     (load_workspace_uw, UnweightedHypergraph, super::descriptors::WorkspaceStdUnweightedLoader),
//     (load_workspace_w, WeightedHypergraph, super::descriptors::WorkspaceStdWeightedLoader),
//     (load_dblp_uw, UnweightedHypergraph, super::descriptors::DblpStdUnweightedLoader),
//     (load_dblp_w, WeightedHypergraph, super::descriptors::DblpStdWeightedLoader),
//     (load_history_uw, UnweightedHypergraph, super::descriptors::HistoryStdUnweightedLoader),
//     (load_history_w, WeightedHypergraph, super::descriptors::HistoryStdWeightedLoader),
//     (load_geology_uw, UnweightedHypergraph, super::descriptors::GeologyStdUnweightedLoader),
//     (load_geology_w, WeightedHypergraph, super::descriptors::GeologyStdWeightedLoader),
//     (load_justice_uw, UnweightedHypergraph, super::descriptors::JusticeStdUnweightedLoader),
//     (load_justice_w, WeightedHypergraph, super::descriptors::JusticeStdWeightedLoader),
//     // (load_babbuini_uw, UnweightedHypergraph, super::babbuini::Unweighted),
//     // (load_babbuini_w, WeightedHypergraph, super::babbuini::Weighted),
//     (load_wiki_uw, UnweightedHypergraph, super::descriptors::WikiStdUnweightedLoader),
//     (load_wiki_w, WeightedHypergraph, super::descriptors::WikiStdWeightedLoader),
//     (load_ndc_substances_uw, UnweightedHypergraph, super::descriptors::NdcSubstancesStdUnweightedLoader),
//     (load_ndc_substances_w, WeightedHypergraph, super::descriptors::NdcSubstancesStdWeightedLoader),
//     (load_ndc_classes_uw, UnweightedHypergraph, super::descriptors::NdcClassesStdUnweightedLoader),
//     (load_ndc_classes_w, WeightedHypergraph, super::descriptors::NdcClassesStdWeightedLoader),
//     (load_eu_uw, UnweightedHypergraph, super::descriptors::EuStdUnweightedLoader),
//     (load_eu_w, WeightedHypergraph, super::descriptors::EuStdWeightedLoader),
//     (load_enron_uw, UnweightedHypergraph, super::descriptors::EnronStdUnweightedLoader),
//     (load_enron_w, WeightedHypergraph, super::descriptors::EnronStdWeightedLoader),
//     (load_wiki_talk_2_uniform, AdjList, super::wiki_talk::Unweighted2Uniform)
// );

use crate::loader::common::DatasetInfo;
use crate::loader::common::Loader;
use better_default::Default;
use pyo3::PyResult;
use pyo3::exceptions::PyIOError;
use pyo3::{pyclass, pymethods};
use pyo3_stub_gen::reexport_module_members;
use rust_core_macros::hoist_mod;
use rust_core_macros::loaders;
use std::error::Error;
use std::path::PathBuf;

// #[pyclass]
// #[derive(Clone)]
// struct X {}
//
// #[pymethods]
// impl X {
//     fn new(&self) -> Self {
//         self.clone()
//     }
// }

#[loaders]
#[hoist_mod]
mod __ {

    // Main dispatcher
    #[loader]
    pub struct CommonLoader {
        #[default(true)]
        cached: bool,

        #[builder::skip]
        dataset_location: PathBuf,
        #[builder::skip]
        cache_dir: Option<PathBuf>,
    }

    // Hospital
    #[subloader(CommonLoader)]
    pub struct HospitalCommonLoader {}

    #[subloader(HospitalCommonLoader)]
    pub struct HospitalStdWeightedLoader {}

    #[subloader(HospitalCommonLoader)]
    pub struct HospitalStdUnweightedLoader {}

    // Conference
    #[subloader(CommonLoader)]
    pub struct ConferenceCommonLoader {}

    #[subloader(ConferenceCommonLoader)]
    pub struct ConferenceStdUnweightedLoader {}

    #[subloader(ConferenceCommonLoader)]
    pub struct ConferenceStdWeightedLoader {}

    // Primary school
    #[subloader(CommonLoader)]
    pub struct PrimarySchoolCommonLoader {}

    #[subloader(PrimarySchoolCommonLoader)]
    pub struct PrimarySchoolStdUnweightedLoader {}

    #[subloader(PrimarySchoolCommonLoader)]
    pub struct PrimarySchoolStdWeightedLoader {}

    // High school
    #[subloader(CommonLoader)]
    pub struct HighSchoolCommonLoader {}

    #[subloader(HighSchoolCommonLoader)]
    pub struct HighSchoolStdUnweightedLoader {}

    #[subloader(HighSchoolCommonLoader)]
    pub struct HighSchoolStdWeightedLoader {}

    // Facebook HS
    #[subloader(CommonLoader)]
    pub struct FacebookHsCommonLoader {}

    #[subloader(FacebookHsCommonLoader)]
    pub struct FacebookHsStdUnweightedLoader {}

    // #[subloader(FacebookHsCommonLoader)]
    // pub struct FacebookHsStdWeightedLoader {}

    // Friendship HS
    #[subloader(CommonLoader)]
    pub struct FriendshipHsCommonLoader {}

    #[subloader(FriendshipHsCommonLoader)]
    pub struct FriendshipHsStdUnweightedLoader {}

    #[subloader(FriendshipHsCommonLoader)]
    pub struct FriendshipHsStdWeightedLoader {}

    // Gene disease
    #[subloader(CommonLoader)]
    pub struct GeneDiseaseCommonLoader {}

    #[subloader(GeneDiseaseCommonLoader)]
    pub struct GeneDiseaseStdWeightedLoader {}

    // PACS
    #[subloader(CommonLoader)]
    pub struct PacsCommonLoader {}

    #[subloader(PacsCommonLoader)]
    pub struct PacsStdUnweightedLoader {}

    #[subloader(PacsCommonLoader)]
    pub struct PacsStdWeightedLoader {}

    // Workspace
    #[subloader(CommonLoader)]
    pub struct WorkspaceCommonLoader {}

    #[subloader(WorkspaceCommonLoader)]
    pub struct WorkspaceStdUnweightedLoader {}

    #[subloader(WorkspaceCommonLoader)]
    pub struct WorkspaceStdWeightedLoader {}

    // DBLP
    #[subloader(CommonLoader)]
    pub struct DblpCommonLoader {}

    #[subloader(DblpCommonLoader)]
    pub struct DblpStdUnweightedLoader {}

    #[subloader(DblpCommonLoader)]
    pub struct DblpStdWeightedLoader {}

    // History
    #[subloader(CommonLoader)]
    pub struct HistoryCommonLoader {}

    #[subloader(HistoryCommonLoader)]
    pub struct HistoryStdUnweightedLoader {}

    #[subloader(HistoryCommonLoader)]
    pub struct HistoryStdWeightedLoader {}

    // Geology
    #[subloader(CommonLoader)]
    pub struct GeologyCommonLoader {}

    #[subloader(GeologyCommonLoader)]
    pub struct GeologyStdUnweightedLoader {}

    #[subloader(GeologyCommonLoader)]
    pub struct GeologyStdWeightedLoader {}

    // Justice
    #[subloader(CommonLoader)]
    pub struct JusticeCommonLoader {}

    #[subloader(JusticeCommonLoader)]
    pub struct JusticeStdUnweightedLoader {}

    #[subloader(JusticeCommonLoader)]
    pub struct JusticeStdWeightedLoader {}

    // NDC substances
    #[subloader(CommonLoader)]
    pub struct NdcSubstancesCommonLoader {}

    #[subloader(NdcSubstancesCommonLoader)]
    pub struct NdcSubstancesStdUnweightedLoader {}

    #[subloader(NdcSubstancesCommonLoader)]
    pub struct NdcSubstancesStdWeightedLoader {}

    // NDC classes
    #[subloader(CommonLoader)]
    pub struct NdcClassesCommonLoader {}

    #[subloader(NdcClassesCommonLoader)]
    pub struct NdcClassesStdUnweightedLoader {}

    #[subloader(NdcClassesCommonLoader)]
    pub struct NdcClassesStdWeightedLoader {}

    // EU
    #[subloader(CommonLoader)]
    pub struct EuCommonLoader {}

    #[subloader(EuCommonLoader)]
    pub struct EuStdUnweightedLoader {}

    #[subloader(EuCommonLoader)]
    pub struct EuStdWeightedLoader {}

    // Enron
    #[subloader(CommonLoader)]
    pub struct EnronCommonLoader {}

    #[subloader(EnronCommonLoader)]
    pub struct EnronStdUnweightedLoader {}

    #[subloader(EnronCommonLoader)]
    pub struct EnronStdWeightedLoader {}

    // Wiki
    #[subloader(CommonLoader)]
    pub struct WikiCommonLoader {}

    #[subloader(WikiCommonLoader)]
    pub struct WikiStdUnweightedLoader {}

    #[subloader(WikiCommonLoader)]
    pub struct WikiStdWeightedLoader {}
}
pub fn test() {
    let x = CommonLoader::builder()
        .high_school_common_loader()
        .high_school_std_weighted_loader()
        .load();
}

// reexport_module_members!("rust_core.loader" from "rust_core.core.loader");
