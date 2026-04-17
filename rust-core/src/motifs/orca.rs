use pyo3::pyfunction;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

#[pyfunction]
#[gen_stub_pyfunction(module = "rust_core.core.motifs")]
pub fn orca() {
    println!("Orca motifs");
}
