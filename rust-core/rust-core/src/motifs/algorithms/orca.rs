use crate::graph::{UnweightedHypergraph, WeightedHypergraph};
use pyo3::pyfunction;
use pyo3_stub_gen::derive::gen_stub_pyfunction;



pub fn orca_unweighted(hg: &UnweightedHypergraph) {}

pub fn orca_weighted(hg: &WeightedHypergraph) {}
