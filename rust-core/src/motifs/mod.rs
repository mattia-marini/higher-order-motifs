use pyo3::prelude::*;
use pyo3_stub_gen::reexport_module_members;
pub mod base;
pub mod motifs3;
pub mod orca;

#[pymodule(submodule)]
pub mod motifs {
    // #[pymodule_export]
    // use super::motifs3::count_motifs_3;

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.motifs")]
    pub fn count_motifs_3(edges: (Vec<(NodeId, NodeId)>, Vec<(NodeId, NodeId, NodeId)>)) {
        super::motifs3::count_motifs_3(&edges)
        // println!("count_motifs_3 called with edges: {:?}", edges);
    }

    #[pyfunction]
    #[gen_stub_pyfunction(module = "rust_core.core.motifs")]
    pub fn count_motifs_4(
        edges: (
            Vec<(usize, usize)>,
            Vec<(usize, usize, usize)>,
            Vec<(usize, usize, usize, usize)>,
        ),
    ) {
        super::motifs3::count_motifs_4(&edges);
        // println!("count_motifs_4 called with edges: {:?}", edges);
    }

    use pyo3::pyfunction;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;
    use crate::graph::types::NodeId;
    #[pymodule_export]
    use super::orca::orca;
}

reexport_module_members!("rust_core.motifs" from "rust_core.core.motifs");
