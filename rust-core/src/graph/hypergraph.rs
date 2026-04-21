use std::{collections::HashMap, fs::File, io::Write, path::Path, time::Instant};

use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rkyv::{Archive, Deserialize, Serialize};

use super::types::*;
use pyo3::{pyclass, pymethods};

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)] // <--- Add this
#[gen_stub_pyclass(module = "rust_core.core.graph")]
#[pyclass]
pub struct Hypergraph {
    pub h2: Vec<H2>,
    pub h3: Vec<H3>,
    pub h4: Vec<H4>,
    pub h5: Vec<H5>,

    pub bigger_edges: HashMap<usize, Vec<Vec<NodeId>>>,

    n: usize,
    m: usize,
}

#[gen_stub_pymethods(module = "rust_core.core.graph")]
#[pymethods]
impl Hypergraph {
    #[new]
    pub fn new() -> Self {
        Hypergraph {
            h2: vec![],
            h3: vec![],
            h4: vec![],
            h5: vec![],
            bigger_edges: HashMap::new(),
            n: 0,
            m: 0,
        }
    }

    #[inline(always)]
    pub fn add_edge(&mut self, edge: Vec<NodeId>) {
        let size = edge.len();
        match size {
            2 => self.h2.push(H2::new(edge[0], edge[1])),
            3 => self.h3.push(H3::new(edge[0], edge[1], edge[2])),
            4 => self.h4.push(H4::new(edge[0], edge[1], edge[2], edge[3])),
            5 => self
                .h5
                .push(H5::new(edge[0], edge[1], edge[2], edge[3], edge[4])),
            _ => self.bigger_edges.entry(size).or_insert(vec![]).push(edge),
        }
    }

    #[inline(always)]
    pub fn add_h2(&mut self, edge: H2) {
        self.h2.push(edge);
        self.m += 1;
    }
    #[inline(always)]
    pub fn extends_h2(&mut self, edges: Vec<H2>) {
        self.m += edges.len();
        self.h2.extend(edges);
    }

    #[inline(always)]
    pub fn add_h3(&mut self, edge: H3) {
        self.h3.push(edge);
        self.m += 1;
    }
    #[inline(always)]
    pub fn extends_h3(&mut self, edges: Vec<H3>) {
        self.m += edges.len();
        self.h3.extend(edges);
    }

    #[inline(always)]
    pub fn add_h4(&mut self, edge: H4) {
        self.h4.push(edge);
        self.m += 1;
    }
    #[inline(always)]
    pub fn extends_h4(&mut self, edges: Vec<H4>) {
        self.m += edges.len();
        self.h4.extend(edges);
    }

    #[inline(always)]
    pub fn add_h5(&mut self, edge: H5) {
        self.h5.push(edge);
        self.m += 1;
    }
    #[inline(always)]
    pub fn extends_h5(&mut self, edges: Vec<H5>) {
        self.m += edges.len();
        self.h5.extend(edges);
    }
}

impl Hypergraph {
    pub fn from_edges(edges: Vec<Vec<NodeId>>) -> Self {
        let mut hg = Hypergraph::new();
        for edge in edges {
            let size = edge.len();
            match size {
                2 => hg.h2.push(H2::new(edge[0], edge[1])),
                3 => hg.h3.push(H3::new(edge[0], edge[1], edge[2])),
                4 => hg.h4.push(H4::new(edge[0], edge[1], edge[2], edge[3])),
                5 => hg
                    .h5
                    .push(H5::new(edge[0], edge[1], edge[2], edge[3], edge[4])),
                _ => hg.bigger_edges.entry(size).or_insert(vec![]).push(edge),
            }
        }
        hg
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self)?;

        let mut file = File::create(path)?;
        file.write_all(&bytes)?;

        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<Hypergraph, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;

        let start = Instant::now();
        let archived = rkyv::access::<ArchivedHypergraph, rkyv::rancor::Error>(&bytes[..])?;
        // let archived = unsafe { rkyv::access_unchecked::<ArchivedFlatAdjList>(&bytes[..]) }; // Faster
        println!("Loading ArchivedHypergraph {:?}", start.elapsed());

        let start = Instant::now();
        let rv = rkyv::deserialize::<Hypergraph, rkyv::rancor::Error>(archived)?;
        println!("Loading Hypergraph{:?}", start.elapsed());

        Ok(rv)
    }
}
