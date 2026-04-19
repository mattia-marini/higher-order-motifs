use std::collections::HashMap;

use super::types::*;
use pyo3::{pyclass, pymethods};

#[pyclass]
pub struct Hypergraph {
    h2: Vec<H2>,
    h3: Vec<H3>,
    h4: Vec<H4>,
    h5: Vec<H5>,

    bigger_edges: HashMap<usize, Vec<Vec<NodeId>>>,

    n: usize,
    m: usize,
}

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
}
