use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Write,
    path::Path,
    time::Instant,
};

use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rkyv::{Archive, Deserialize, Serialize};

use super::types::*;
use pyo3::{pyclass, pymethods};

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
#[pyclass]
pub struct UnweightedHypergraph {
    pub h2: Vec<H2>,
    pub h3: Vec<H3>,
    pub h4: Vec<H4>,
    pub h5: Vec<H5>,

    pub bigger_edges: HashMap<usize, Vec<Vec<NodeId>>>,
    pub nodes: HashSet<NodeId>,

    #[pyo3(get)]
    n: usize,
    #[pyo3(get)]
    m: usize,
}

#[gen_stub_pymethods(module = "rust_core.core.graph")]
#[pymethods]
impl UnweightedHypergraph {
    #[new]
    pub fn new() -> Self {
        UnweightedHypergraph {
            h2: vec![],
            h3: vec![],
            h4: vec![],
            h5: vec![],
            bigger_edges: HashMap::new(),
            nodes: HashSet::new(),
            n: 0,
            m: 0,
        }
    }

    #[inline(always)]
    pub fn add_edge(&mut self, edge: Vec<NodeId>) {
        let size = edge.len();
        edge.iter().for_each(|&node| {
            if self.nodes.insert(node) {
                self.n += 1;
            }
        });

        match size {
            2 => self.h2.push((edge[0], edge[1])),
            3 => self.h3.push((edge[0], edge[1], edge[2])),
            4 => self.h4.push((edge[0], edge[1], edge[2], edge[3])),
            5 => self.h5.push((edge[0], edge[1], edge[2], edge[3], edge[4])),
            _ => self.bigger_edges.entry(size).or_insert(vec![]).push(edge),
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

    #[inline(always)]
    pub fn add_h2(&mut self, edge: H2) {
        for node in &[edge.0, edge.1] {
            if self.nodes.insert(*node) {
                self.n += 1;
            }
        }
        self.h2.push(edge);
        self.m += 1;
    }
    #[inline(always)]
    pub fn extends_h2(&mut self, edges: Vec<H2>) {
        for edge in &edges {
            for node in &[edge.0, edge.1] {
                if self.nodes.insert(*node) {
                    self.n += 1;
                }
            }
        }
        self.m += edges.len();
        self.h2.extend(edges);
    }

    #[inline(always)]
    pub fn add_h3(&mut self, edge: H3) {
        for node in &[edge.0, edge.1, edge.2] {
            if self.nodes.insert(*node) {
                self.n += 1;
            }
        }
        self.h3.push(edge);
        self.m += 1;
    }
    #[inline(always)]
    pub fn extends_h3(&mut self, edges: Vec<H3>) {
        for edge in &edges {
            for node in &[edge.0, edge.1, edge.2] {
                if self.nodes.insert(*node) {
                    self.n += 1;
                }
            }
        }
        self.m += edges.len();
        self.h3.extend(edges);
    }

    #[inline(always)]
    pub fn add_h4(&mut self, edge: H4) {
        for node in &[edge.0, edge.1, edge.2, edge.3] {
            if self.nodes.insert(*node) {
                self.n += 1;
            }
        }
        self.h4.push(edge);
        self.m += 1;
    }
    #[inline(always)]
    pub fn extends_h4(&mut self, edges: Vec<H4>) {
        for edge in &edges {
            for node in &[edge.0, edge.1, edge.2, edge.3] {
                if self.nodes.insert(*node) {
                    self.n += 1;
                }
            }
        }
        self.m += edges.len();
        self.h4.extend(edges);
    }

    #[inline(always)]
    pub fn add_h5(&mut self, edge: H5) {
        for node in &[edge.0, edge.1, edge.2, edge.3, edge.4] {
            if self.nodes.insert(*node) {
                self.n += 1;
            }
        }
        self.h5.push(edge);
        self.m += 1;
    }
    #[inline(always)]
    pub fn extends_h5(&mut self, edges: Vec<H5>) {
        for edge in &edges {
            for node in &[edge.0, edge.1, edge.2, edge.3, edge.4] {
                if self.nodes.insert(*node) {
                    self.n += 1;
                }
            }
        }
        self.m += edges.len();
        self.h5.extend(edges);
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

impl UnweightedHypergraph {
    pub fn from_edges(edges: Vec<Vec<NodeId>>) -> Self {
        let mut hg = UnweightedHypergraph::new();
        for edge in edges {
            let size = edge.len();
            match size {
                2 => hg.h2.push((edge[0], edge[1])),
                3 => hg.h3.push((edge[0], edge[1], edge[2])),
                4 => hg.h4.push((edge[0], edge[1], edge[2], edge[3])),
                5 => hg.h5.push((edge[0], edge[1], edge[2], edge[3], edge[4])),
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
    ) -> Result<UnweightedHypergraph, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;

        // let start = Instant::now();
        let archived =
            rkyv::access::<ArchivedUnweightedHypergraph, rkyv::rancor::Error>(&bytes[..])?;
        // let archived = unsafe { rkyv::access_unchecked::<ArchivedFlatAdjList>(&bytes[..]) }; // Faster
        // println!("Loading ArchivedHypergraph {:?}", start.elapsed());

        // let start = Instant::now();
        let rv = rkyv::deserialize::<UnweightedHypergraph, rkyv::rancor::Error>(archived)?;
        // println!("Loading Hypergraph{:?}", start.elapsed());

        Ok(rv)
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
#[pyclass]
pub struct WeightedHypergraph {
    pub h2: Vec<WH2>,
    pub h3: Vec<WH3>,
    pub h4: Vec<WH4>,
    pub h5: Vec<WH5>,

    // Maps edge size to a list of (Weight, Nodes)
    pub bigger_edges: HashMap<usize, Vec<(NodeWeight, Vec<NodeId>)>>,
    pub nodes: HashSet<NodeId>,

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
            nodes: HashSet::new(),
            n: 0,
            m: 0,
        }
    }

    #[inline(always)]
    pub fn add_edge(&mut self, weight: NodeWeight, nodes: Vec<NodeId>) {
        let size = nodes.len();
        nodes.iter().for_each(|&node| {
            if self.nodes.insert(node) {
                self.n += 1;
            }
        });

        match size {
            2 => self.h2.push((weight, (nodes[0], nodes[1]))),
            3 => self.h3.push((weight, (nodes[0], nodes[1], nodes[2]))),
            4 => self
                .h4
                .push((weight, (nodes[0], nodes[1], nodes[2], nodes[3]))),
            5 => self
                .h5
                .push((weight, (nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]))),
            _ => self
                .bigger_edges
                .entry(size)
                .or_insert(vec![])
                .push((weight, nodes)),
        }
        self.m += 1;
    }

    // --- Specialized Adders ---

    #[inline(always)]
    pub fn add_wh2(&mut self, edge: WH2) {
        let (_, (n1, n2)) = edge;
        for node in &[n1, n2] {
            if self.nodes.insert(*node) {
                self.n += 1;
            }
        }
        self.h2.push(edge);
        self.m += 1;
    }

    #[inline(always)]
    pub fn extends_wh2(&mut self, edges: Vec<WH2>) {
        for edge in &edges {
            let (_, (n1, n2)) = edge;
            for node in &[*n1, *n2] {
                if self.nodes.insert(*node) {
                    self.n += 1;
                }
            }
        }
        self.m += edges.len();
        self.h2.extend(edges);
    }

    #[inline(always)]
    pub fn add_wh3(&mut self, edge: WH3) {
        let (_, (n1, n2, n3)) = edge;
        for node in &[n1, n2, n3] {
            if self.nodes.insert(*node) {
                self.n += 1;
            }
        }
        self.h3.push(edge);
        self.m += 1;
    }

    #[inline(always)]
    pub fn extends_wh3(&mut self, edges: Vec<WH3>) {
        for edge in &edges {
            let (_, (n1, n2, n3)) = edge;
            for node in &[*n1, *n2, *n3] {
                if self.nodes.insert(*node) {
                    self.n += 1;
                }
            }
        }
        self.m += edges.len();
        self.h3.extend(edges);
    }

    // --- Stats ---

    pub fn count_2(&self) -> usize {
        self.h2.len()
    }
    pub fn count_3(&self) -> usize {
        self.h3.len()
    }
    pub fn m(&self) -> usize {
        self.m
    }
    pub fn n(&self) -> usize {
        self.n
    }
}

impl WeightedHypergraph {
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
