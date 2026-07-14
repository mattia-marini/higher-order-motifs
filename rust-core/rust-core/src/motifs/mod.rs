#[cfg(feature = "bindings")]
use pyo3::prelude::*;

#[cfg(feature = "bindings")]
use pyo3_stub_gen::{PyStubType, impl_stub_type, reexport_module_members};

pub mod algorithms;
pub mod compressed_motif;
pub mod compressed_node_set;
pub mod fingerprint;
pub mod types;

#[cfg(feature = "bindings")]
#[pymodule]
pub mod motifs {
    use hashbrown::HashMap;

    use crate::{
        motifs::{
            fingerprint::{Fingerprint3, Fingerprint4},
            types::MotifStats,
        },
        types::{PyHypergraph, hyperadj_list::HyperAdjList},
    };
    use pyo3::pyfunction;
    use pyo3_stub_gen::{derive::gen_stub_pyfunction, reexport_module_members};

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    pub fn analyze_esu_based_3(hg: PyHypergraph) {
        match hg {
            PyHypergraph::Unweighted(unweighted) => {
                super::algorithms::esu_based::unweighted_3(&unweighted);
            }
            PyHypergraph::Weighted(weighted) => {
                super::algorithms::esu_based::weighted_3(&weighted);
            }
        }
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    pub fn analyze_esu_based_4(hg: PyHypergraph) {
        match hg {
            PyHypergraph::Unweighted(unweighted) => {
                super::algorithms::esu_based::unweighted_4(&unweighted);
            }
            PyHypergraph::Weighted(weighted) => {
                super::algorithms::esu_based::weighted_4(&weighted);
            }
        }
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    pub fn orca_3(hg: PyHypergraph) -> HashMap<Fingerprint3, MotifStats> {
        match hg {
            PyHypergraph::Unweighted(unweighted) => {
                super::algorithms::orca::unweighted_3(&unweighted)
            }
            PyHypergraph::Weighted(weighted) => super::algorithms::orca::weighted_3(&weighted),
        }
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    pub fn orca_4(hg: PyHypergraph) -> HashMap<Fingerprint4, MotifStats> {
        match hg {
            PyHypergraph::Unweighted(unweighted) => {
                super::algorithms::orca::unweighted_4(&unweighted)
            }
            PyHypergraph::Weighted(weighted) => super::algorithms::orca::weighted_4(&weighted),
        }
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    pub fn escape_3(hg: PyHypergraph) -> HashMap<Fingerprint3, MotifStats> {
        todo!()
        // match hg {
        //     PyHypergraph::Unweighted(unweighted) => {
        //         super::algorithms::escape::unweighted_4(&unweighted)
        //     }
        //     PyHypergraph::Weighted(weighted) => super::algorithms::orca::weighted_4(&weighted),
        // }
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    pub fn escape_4(hg: PyHypergraph) -> HashMap<Fingerprint4, MotifStats> {
        match hg {
            PyHypergraph::Unweighted(unweighted) => {
                let (adj, _, _) = HyperAdjList::from_hypergraph_mapped(unweighted.0.clone());
                super::algorithms::escape::unweighted_4(&adj)
            }
            PyHypergraph::Weighted(weighted) => {
                let (adj, _, _) = HyperAdjList::from_hypergraph_mapped(weighted.0.clone());
                super::algorithms::escape::weighted_4(&adj)
            }
        }
    }

    // #[pyfunction]
    // #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    // pub fn orca_5(hg: PyHypergraph) {
    //     match hg {
    //         PyHypergraph::Unweighted(unweighted) => {
    //             super::algorithms::orca::unweighted_5(&unweighted);
    //         }
    //         PyHypergraph::Weighted(weighted) => {
    //             super::algorithms::orca::weighted_5(&weighted);
    //         }
    //     }
    // }
    //
    reexport_module_members!("rust_core.motifs" from "rust_core._core.motifs");
}
