use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Write,
    path::Path,
};

use pyo3_stub_gen::{
    PyStubType,
    derive::{gen_stub_pyclass, gen_stub_pymethods},
    impl_stub_type, type_alias,
};
use rkyv::{Archive, Deserialize, Serialize, collections::swiss_table::ArchivedHashSet};

use super::types::*;
use pyo3::{Bound, FromPyObject, PyRef, PyResult, pyclass, pymethods};

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
#[pyclass]
pub struct WeightedHypergraph {
    pub h2: Vec<WH2>,
    pub h3: Vec<WH3>,
    pub h4: Vec<WH4>,
    pub h5: Vec<WH5>,

    pub bigger_edges: HashMap<usize, Vec<WHx>>,

    pub edges: HashSet<Hx>,
    pub nodes: HashMap<NodeId, usize>, // track number of edges insisting on a certain node

    #[pyo3(get)]
    n: usize,
    #[pyo3(get)]
    m: usize,
}

#[gen_stub_pymethods(module = "rust_core.core.graph")]
#[pymethods]
impl WeightedHypergraph {
    #[new]
    pub fn new() -> Self {
        WeightedHypergraph {
            h2: vec![],
            h3: vec![],
            h4: vec![],
            h5: vec![],
            bigger_edges: HashMap::new(),
            nodes: HashMap::new(),
            edges: HashSet::new(),
            n: 0,
            m: 0,
        }
    }

    #[inline(always)]
    pub fn add_edge(&mut self, edge: WHx) {
        let hx = edge.edge.clone();

        if !self.edges.contains(&hx) {
            self.update_n(&hx);
            self.edges.insert(hx);

            let size = edge.len();
            match size {
                2 => self
                    .h2
                    .push(WH2::new_unchecked(edge.weight, edge[0], edge[1])),
                3 => self
                    .h3
                    .push(WH3::new_unchecked(edge.weight, edge[0], edge[1], edge[2])),
                4 => self.h4.push(WH4::new_unchecked(
                    edge.weight,
                    edge[0],
                    edge[1],
                    edge[2],
                    edge[3],
                )),
                5 => self.h5.push(WH5::new_unchecked(
                    edge.weight,
                    edge[0],
                    edge[1],
                    edge[2],
                    edge[3],
                    edge[4],
                )),
                _ => self.bigger_edges.entry(size).or_insert(vec![]).push(edge),
            }
            self.m += 1;
        }
    }

    pub fn extend_with_edges(&mut self, edges: Vec<WHx>) {
        for edge in edges.into_iter() {
            self.add_edge(edge);
        }
    }

    #[inline(always)]
    pub fn add_h2(&mut self, edge: WH2) {
        let hx = edge.nodes.clone().into();
        if !self.edges.contains(&hx) {
            self.update_n(&hx);

            self.edges.insert(hx.into());
            self.h2.push(edge);
            self.m += 1;
        }
    }

    #[inline(always)]
    pub fn add_h3(&mut self, edge: WH3) {
        let hx = edge.nodes.clone().into();
        if !self.edges.contains(&hx) {
            self.update_n(&hx);

            self.edges.insert(hx.into());
            self.h3.push(edge);
            self.m += 1;
        }
    }

    #[inline(always)]
    pub fn add_h4(&mut self, edge: WH4) {
        let hx = edge.nodes.clone().into();
        if !self.edges.contains(&hx) {
            self.update_n(&hx);

            self.edges.insert(hx.into());
            self.h4.push(edge);
            self.m += 1;
        }
    }

    #[inline(always)]
    pub fn add_h5(&mut self, edge: WH5) {
        let hx = edge.nodes.clone().into();
        if !self.edges.contains(&hx) {
            self.update_n(&hx);

            self.edges.insert(hx.into());
            self.h5.push(edge);
            self.m += 1;
        }
    }

    pub fn extends_wh2(&mut self, edges: Vec<WH2>) {
        for edge in edges {
            self.add_h2(edge)
        }
    }

    pub fn extends_wh3(&mut self, edges: Vec<WH3>) {
        for edge in edges {
            self.add_h3(edge)
        }
    }

    pub fn extends_wh4(&mut self, edges: Vec<WH4>) {
        for edge in edges {
            self.add_h4(edge)
        }
    }

    pub fn extends_wh5(&mut self, edges: Vec<WH5>) {
        for edge in edges {
            self.add_h5(edge)
        }
    }

    pub fn count_2(&self) -> usize {
        self.h2.len()
    }

    pub fn count_3(&self) -> usize {
        self.h3.len()
    }

    pub fn count_4(&self) -> usize {
        self.h4.len()
    }

    pub fn count_5(&self) -> usize {
        self.h5.len()
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

impl WeightedHypergraph {
    #[inline(always)]
    fn update_n<T>(&mut self, edge: &T)
    where
        for<'a> &'a T: IntoIterator<Item = &'a NodeId>,
    {
        for node in edge {
            self.nodes
                .entry(*node)
                .and_modify(|count| *count += 1)
                .or_insert_with(|| {
                    self.n += 1;
                    1
                });
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self)?;
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<WeightedHypergraph, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let archived = rkyv::access::<ArchivedWeightedHypergraph, rkyv::rancor::Error>(&bytes[..])?;
        let rv = rkyv::deserialize::<WeightedHypergraph, rkyv::rancor::Error>(archived)?;
        Ok(rv)
    }
}
