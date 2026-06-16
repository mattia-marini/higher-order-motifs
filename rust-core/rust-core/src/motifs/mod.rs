use pyo3::prelude::*;
use pyo3_stub_gen::{PyStubType, impl_stub_type, reexport_module_members};
pub mod algorithms;
pub mod compressed_motif;
pub mod compressed_node_set;
pub mod fingerprint;
pub mod types;

#[pymodule(submodule)]
pub mod motifs {
    use hashbrown::HashMap;

    use crate::{
        graph::{PyHypergraph, UnweightedHypergraph, WeightedHypergraph},
        motifs::{fingerprint::Fingerprint3, types::MotifStats},
    };
    use pyo3::pyfunction;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    /// Computed
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
    pub fn orca_4(hg: PyHypergraph) -> HashMap<u32, u32> {
        // match hg {
        //     PyHypergraph::Unweighted(unweighted) => {
        //         super::algorithms::orca::unweighted_4(&unweighted);
        //     }
        //     PyHypergraph::Weighted(weighted) => {
        //         super::algorithms::orca::weighted_4(&weighted);
        //     }
        // }
        todo!()
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core._core.motifs")]
    pub fn orca_5(hg: PyHypergraph) {
        match hg {
            PyHypergraph::Unweighted(unweighted) => {
                super::algorithms::orca::unweighted_5(&unweighted);
            }
            PyHypergraph::Weighted(weighted) => {
                super::algorithms::orca::weighted_5(&weighted);
            }
        }
    }
}

// impl_stub_type!(hashbrown::HashMap<_, _> = std::collections::HashMap<_, _>);
// impl<K, V> PyStubType for hashbrown::HashMap<K, V> {
//     fn type_output() -> ::pyo3_stub_gen::TypeInfo {
//         std::collections::HashMap::<K, V>::type_output()
//     }
//     fn type_input() -> ::pyo3_stub_gen::TypeInfo {
//         std::collections::HashMap::<K, V>::type_input()
//     }
// }

reexport_module_members!("rust_core.motifs" from "rust_core._core.motifs");
