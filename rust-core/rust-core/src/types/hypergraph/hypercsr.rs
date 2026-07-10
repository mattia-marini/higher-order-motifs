use foldhash::fast::FixedState;
use hashbrown::HashSet;
use num_traits::AsPrimitive;
use std::hash::Hash;

use crate::types::{EdgeId, Hypergraph, NodeId};

#[derive(Clone)]
pub struct HyperCSR<W> {
    pub(crate) nodes: Vec<NodeId>,
    pub(crate) weights: Vec<W>,
    pub(crate) n: usize,
    pub(crate) m: usize,

    /// To allow iterating by hyperedge size; sizes[i] = (start_index, count) of edges of size i
    pub(crate) sizes: Vec<(usize, usize)>,

    /// Lookup table; lookup[e] = inxed of first node of edge e in self.nodes
    pub(crate) lookup: Vec<(usize, u8)>,
}

pub struct EdgeRef<'a, W> {
    pub nodes: &'a [NodeId],
    pub weight: &'a W,
}

pub struct EdgeRefMut<'a, W> {
    pub nodes: &'a [NodeId],
    pub weight: &'a mut W,
}

impl<W> HyperCSR<W> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            weights: Vec::new(),
            n: 0,
            m: 0,
            sizes: Vec::new(),
            lookup: Vec::new(),
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn from_hypergraph<T: Hash + Eq + AsPrimitive<NodeId>>(mut hg: Hypergraph<T, W>) -> Self {
        let mut rv = Self::new();
        let mut edge_id = 0;
        let mut edge_pos = 0;
        rv.lookup.reserve(hg.m());
        rv.weights.reserve(hg.m());
        rv.sizes.reserve(11);
        rv.m = hg.m();
        rv.n = hg.n();

        seq_macro::seq!(N in 2..11{
            rv.sizes.push((edge_id, hg.edges::<N>().len()));
            for edge in hg.into_iter_edges::<N>() {
                rv.lookup.push((edge_pos, N));
                for n in &edge{
                    rv.nodes.push(n.as_());
                }

                rv.weights.push(edge.weight);
                edge_id += 1;
                edge_pos += N;
            }
        });
        rv
    }

    pub fn iter_by_size(&self, size: usize) -> impl Iterator<Item = (EdgeId, EdgeRef<'_, W>)> + '_ {
        let (first_id, count) = self.sizes[size];
        let start = self.lookup[first_id].0;

        (0..count).map(move |number| {
            let edge_id = first_id + number;
            let edge_start = start + size * number;

            let edge_ref = EdgeRef {
                nodes: &self.nodes[edge_start..edge_start + size],
                weight: &self.weights[edge_id],
            };

            (edge_id as NodeId, edge_ref)
        })
    }

    pub fn get_edge_by_id(&self, edge_id: EdgeId) -> EdgeRef<'_, W> {
        let node_start = self.lookup[edge_id as usize].0;
        let edge_size = self.lookup[edge_id as usize].1 as usize;
        EdgeRef {
            nodes: &self.nodes[node_start..node_start + edge_size],
            weight: &self.weights[edge_id as usize],
        }
    }
}
