pub mod common;
pub mod conference;
pub mod dblp;
pub mod enron;
pub mod error;
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

pub use self::loader::*;

use std::error::Error;
use std::path::PathBuf;

#[cfg(feature = "bindings")]
use pyo3::{pyclass, pymethods};
#[cfg(feature = "bindings")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use rust_core_macros::remove_attr;

use rust_core_macros::loaders;

use super::loader::common::parse_datasets_descriptor;

pub fn to_absolute(path: PathBuf) -> PathBuf {
    if path.is_relative() {
        let base =
            std::env::var("DATASETS_TOML").expect("Cannot find DATASETS_TOML environment variable");

        PathBuf::from(base)
            .parent()
            .expect("Invalid DATASETS_TOML path")
            .join(path)
    } else {
        path
    }
}

pub fn get_dataset_loc(dataset_name: &'static str) -> PathBuf {
    let rv = parse_datasets_descriptor().expect("Could not parse datasets.toml");

    let dataset_path = rv
        .datasets
        .get(dataset_name)
        .expect(&format!(
            "Dataset {} not found in datasets.toml",
            dataset_name
        ))
        .path
        .clone();

    let rv = to_absolute(dataset_path);
    rv
}

pub fn get_cache_dir(dataset_name: &'static str) -> Option<PathBuf> {
    let rv = parse_datasets_descriptor().expect("Could not parse datasets.toml");

    if let Some(cache_dir) = rv
        .datasets
        .get(dataset_name)
        .expect(&format!(
            "Dataset {} not found in datasets.toml",
            dataset_name
        ))
        .cache_dir
        .clone()
    {
        return Some(to_absolute(cache_dir));
    }

    return rv.cache_dir.clone().map(|cache_dir| to_absolute(cache_dir));
}

macro_rules! new_loader {
    ($self: ident, $struct_ident: ident, $name: expr) => {{
        let mut rv = $struct_ident::default();
        rv.name = $name;
        rv.dataset_location = super::get_dataset_loc($name);
        rv.cache_dir = super::get_cache_dir($name);
        rv.cached = $self.cached;
        rv
    }};
}

/// Only class exposed to python bindings.
#[cfg_attr(
    feature = "bindings",
    gen_stub_pyclass(module = "rust_core._core.graph"),
    pyclass(from_py_object)
)]
#[derive(Clone)]
pub struct DatasetLoader {}

#[cfg_attr(
    feature = "bindings",
    gen_stub_pymethods(module = "rust_core._core.graph"),
    pymethods
)]
#[cfg_attr(not(feature = "bindings"), remove_attr("staticmethod"))]
impl DatasetLoader {
    #[staticmethod]
    pub fn builder() -> self::loader::DatasetLoaderDispatcher {
        self::loader::DatasetLoaderDispatcher::default()
    }
}

// Mod is removed by hoist_mod; it just provides context for the loaders macro
#[loaders]
#[loaders::struct_attr(derive(Default, Clone, Debug, Hash, PartialEq, Eq))]
#[loaders::struct_attr(cfg_attr(
    feature = "bindings",
    gen_stub_pyclass(module = "rust_core._core.loader"),
    pyclass(from_py_object)
))]
#[loaders::impl_attr(cfg_attr(
    feature = "bindings",
    gen_stub_pymethods(module = "rust_core._core.loader"),
    pymethods
))]
#[cfg_attr(feature = "bindings", pyo3::pymodule)]
pub mod loader {
    use crate::loader::common::DatasetInfo;
    use crate::loader::common::Loader;
    use crate::loader::common::parse_datasets_descriptor;
    use crate::loader::error::LoaderError;
    use better_default::Default;

    #[cfg(feature = "bindings")]
    use pyo3::{PyResult, exceptions::PyIOError, pyclass, pymethods, pymodule};
    #[cfg(feature = "bindings")]
    use pyo3_stub_gen::{
        derive::{gen_stub_pyclass, gen_stub_pymethods},
        exclude_from_all, reexport_module_members,
    };
    use rust_core_macros::hoist_mod;
    use std::error::Error;
    use std::path::Path;
    use std::path::PathBuf;

    // #[cfg(feature = "bindings")]
    // #[loaders::leaf_function]
    // pub fn load(&self) -> Result<<struct_ident as Loader>::Output, LoaderError> {
    //     let dataset_location = self.dataset_location.clone();
    //     <struct_ident as Loader>::load(self)
    //         .map(|v| v.into())
    //         .map_err(|e| {
    //             PyIOError::new_err(format!("Could not load {}", dataset_location.display()))
    //         })
    // }

    #[loaders::leaf_function]
    pub fn load(&self) -> Result<<struct_ident as Loader>::Output, LoaderError> {
        <struct_ident as Loader>::load(self)
    }

    // #[loaders::for_each]
    // exclude_from_all!("rust_core.loader", stringify!(struct_ident));
    //
    // #[loaders::for_each]
    // exclude_from_all!("rust_core._core.loader", stringify!(struct_ident));

    #[loaders::primary]
    pub struct DatasetLoaderDispatcher {
        pub cached: bool,

        #[loaders::builder(skip)]
        pub name: &'static str,
        #[loaders::builder(skip)]
        pub dataset_location: PathBuf,
        #[loaders::builder(skip)]
        pub cache_dir: Option<PathBuf>,
    }

    #[rustfmt::skip]
    impl DatasetLoaderDispatcher {
        pub fn hospital(&self) -> HospitalCommonLoader { new_loader!(self, HospitalCommonLoader, "hospital") }
        pub fn conference(&self) -> ConferenceCommonLoader { new_loader!(self, ConferenceCommonLoader, "conference") }
        pub fn primary_school(&self) -> PrimarySchoolCommonLoader { new_loader!(self, PrimarySchoolCommonLoader, "primary_school") }
        pub fn high_school(&self) -> HighSchoolCommonLoader { new_loader!(self, HighSchoolCommonLoader, "high_school") }
        pub fn facebook_hs(&self) -> FacebookHsCommonLoader { new_loader!(self, FacebookHsCommonLoader, "facebook_hs") }
        pub fn friendship_hs(&self) -> FriendshipHsCommonLoader { new_loader!(self, FriendshipHsCommonLoader, "friendship_hs") }
        pub fn gene_disease(&self) -> GeneDiseaseCommonLoader { new_loader!(self, GeneDiseaseCommonLoader, "gene_disease") }
        pub fn pacs(&self) -> PacsCommonLoader { new_loader!(self, PacsCommonLoader, "pacs") }
        pub fn workspace(&self) -> WorkspaceCommonLoader { new_loader!(self, WorkspaceCommonLoader, "workspace") }
        pub fn dblp(&self) -> DblpCommonLoader { new_loader!(self, DblpCommonLoader, "dblp") }
        pub fn history(&self) -> HistoryCommonLoader { new_loader!(self, HistoryCommonLoader, "history") }
        pub fn geology(&self) -> GeologyCommonLoader { new_loader!(self, GeologyCommonLoader, "geology") }
        pub fn justice(&self) -> JusticeCommonLoader { new_loader!(self, JusticeCommonLoader, "justice") }
        pub fn ndc_substances(&self) -> NdcSubstancesCommonLoader { new_loader!(self, NdcSubstancesCommonLoader, "ndc_substances") }
        pub fn ndc_classes(&self) -> NdcClassesCommonLoader { new_loader!(self, NdcClassesCommonLoader, "ndc_classes") }
        pub fn eu(&self) -> EuCommonLoader { new_loader!(self, EuCommonLoader, "eu") }
        pub fn enron(&self) -> EnronCommonLoader { new_loader!(self, EnronCommonLoader, "enron") }
        pub fn wiki(&self) -> WikiCommonLoader { new_loader!(self, WikiCommonLoader, "wiki") }
    }

    // --- Data Loaders ---

    #[loaders::sub(DatasetLoaderDispatcher, hospital)]
    pub struct HospitalCommonLoader {}
    #[loaders::sub(HospitalCommonLoader, unweighted)]
    pub struct HospitalStdUnweightedLoader {}
    #[loaders::sub(HospitalCommonLoader, weighted)]
    pub struct HospitalStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, conference)]
    pub struct ConferenceCommonLoader {}
    #[loaders::sub(ConferenceCommonLoader, unweighted)]
    pub struct ConferenceStdUnweightedLoader {}
    #[loaders::sub(ConferenceCommonLoader, weighted)]
    pub struct ConferenceStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, primary_school)]
    pub struct PrimarySchoolCommonLoader {}
    #[loaders::sub(PrimarySchoolCommonLoader, unweighted)]
    pub struct PrimarySchoolStdUnweightedLoader {}
    #[loaders::sub(PrimarySchoolCommonLoader, weighted)]
    pub struct PrimarySchoolStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, high_school)]
    pub struct HighSchoolCommonLoader {}
    #[loaders::sub(HighSchoolCommonLoader, unweighted)]
    pub struct HighSchoolStdUnweightedLoader {}
    #[loaders::sub(HighSchoolCommonLoader, weighted)]
    pub struct HighSchoolStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, facebook_hs)]
    pub struct FacebookHsCommonLoader {}
    #[loaders::sub(FacebookHsCommonLoader, unweighted)]
    pub struct FacebookHsStdUnweightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, friendship_hs)]
    pub struct FriendshipHsCommonLoader {}
    #[loaders::sub(FriendshipHsCommonLoader, unweighted)]
    pub struct FriendshipHsStdUnweightedLoader {}
    #[loaders::sub(FriendshipHsCommonLoader, weighted)]
    pub struct FriendshipHsStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, gene_disease)]
    pub struct GeneDiseaseCommonLoader {}
    #[loaders::sub(GeneDiseaseCommonLoader, weighted)]
    pub struct GeneDiseaseStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, pacs)]
    pub struct PacsCommonLoader {}
    #[loaders::sub(PacsCommonLoader, unweighted)]
    pub struct PacsStdUnweightedLoader {}
    #[loaders::sub(PacsCommonLoader, weighted)]
    pub struct PacsStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, workspace)]
    pub struct WorkspaceCommonLoader {}
    #[loaders::sub(WorkspaceCommonLoader, unweighted)]
    pub struct WorkspaceStdUnweightedLoader {}
    #[loaders::sub(WorkspaceCommonLoader, weighted)]
    pub struct WorkspaceStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, dblp)]
    pub struct DblpCommonLoader {}
    #[loaders::sub(DblpCommonLoader, unweighted)]
    pub struct DblpStdUnweightedLoader {}
    #[loaders::sub(DblpCommonLoader, weighted)]
    pub struct DblpStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, history)]
    pub struct HistoryCommonLoader {}
    #[loaders::sub(HistoryCommonLoader, unweighted)]
    pub struct HistoryStdUnweightedLoader {}
    #[loaders::sub(HistoryCommonLoader, weighted)]
    pub struct HistoryStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, geology)]
    pub struct GeologyCommonLoader {}
    #[loaders::sub(GeologyCommonLoader, unweighted)]
    pub struct GeologyStdUnweightedLoader {}
    #[loaders::sub(GeologyCommonLoader, weighted)]
    pub struct GeologyStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, justice)]
    pub struct JusticeCommonLoader {}
    #[loaders::sub(JusticeCommonLoader, unweighted)]
    pub struct JusticeStdUnweightedLoader {}
    #[loaders::sub(JusticeCommonLoader, weighted)]
    pub struct JusticeStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, ndc_substances)]
    pub struct NdcSubstancesCommonLoader {}
    #[loaders::sub(NdcSubstancesCommonLoader, unweighted)]
    pub struct NdcSubstancesStdUnweightedLoader {}
    #[loaders::sub(NdcSubstancesCommonLoader, weighted)]
    pub struct NdcSubstancesStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, ndc_classes)]
    pub struct NdcClassesCommonLoader {}
    #[loaders::sub(NdcClassesCommonLoader, unweighted)]
    pub struct NdcClassesStdUnweightedLoader {}
    #[loaders::sub(NdcClassesCommonLoader, weighted)]
    pub struct NdcClassesStdWeightedLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, eu)]
    pub struct EuCommonLoader {}
    #[loaders::sub(EuCommonLoader, unweighted)]
    pub struct EuStdUnweightedLoader {}
    #[loaders::sub(EuCommonLoader, weighted)]
    pub struct EuStdWeightedLoader {}

    #[loaders::sub(EnronCommonLoader, unweighted)]
    pub struct EnronStdUnweightedLoader {}
    #[loaders::sub(EnronCommonLoader, weighted)]
    pub struct EnronStdWeightedLoader {}
    #[loaders::sub(DatasetLoaderDispatcher, enron)]
    pub struct EnronCommonLoader {}

    #[loaders::sub(DatasetLoaderDispatcher, wiki)]
    pub struct WikiCommonLoader {}
    #[loaders::sub(WikiCommonLoader, unweighted)]
    pub struct WikiStdUnweightedLoader {}
    #[loaders::sub(WikiCommonLoader, weighted)]
    pub struct WikiStdWeightedLoader {}
}

// reexport_module_members!("rust_core.loader" from "rust_core._core.loader");
