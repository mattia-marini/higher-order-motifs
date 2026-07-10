use foldhash::fast::FixedState;
use hashbrown::HashSet;
use num_traits::AsPrimitive;
use std::hash::Hash;

use crate::types::{EdgeId, EdgeRef, HyperCSR, Hypergraph, NodeId};

#[derive(Clone)]
pub struct HyperAdjList<W> {
    pub (crate) csr: HyperCSR<W>,
    pub (crate) adj: Vec<HashSet<usize, FixedState>>,
}

impl<W> HyperAdjList<W> {
    pub fn new() -> Self {
        Self {
            csr: HyperCSR::new(),
            adj: Vec::new(),
        }
    }

    pub fn n(&self) -> usize {
        self.csr.n()
    }

    pub fn m(&self) -> usize {
        self.csr.m()
    }

    pub fn from_hypergraph<T: Hash + Eq + AsPrimitive<NodeId>>(mut hg: Hypergraph<T, W>) -> Self {
        let mut csr = HyperCSR::new();
        let mut edge_pos = 0;
        // csr.lookup.resize(hg.m(), (0, 0));
        csr.lookup.reserve(hg.m());
        csr.weights.reserve(hg.m());
        csr.m = hg.m();
        csr.n = hg.n();

        let mut degrees = vec![0; hg.n()];
        seq_macro::seq!(N in 2..11{
            for edge in hg.into_iter_edges::<N>() {
                csr.lookup.push((edge_pos, N));
                for n in &edge{
                    csr.nodes.push(n.as_());
                    degrees[n.as_() as usize] += 1;
                }
                csr.weights.push(edge.weight);
                edge_pos += N;
            }
        });

        let mut adj = vec![HashSet::with_hasher(FixedState::default()); hg.n()];
        for v in 0..hg.n() {
            adj[v].reserve(degrees[v]);
        }

        for edge_id in 0..csr.m() {
            let edge = csr.get_edge_by_id(edge_id as NodeId);
            for n in edge.nodes {
                adj[*n as usize].insert(edge_id);
            }
        }

        Self { csr, adj }
    }

    /// Returns the set of incident edges ids for a given node.
    pub fn incident_edges(&self, node_id: NodeId) -> &HashSet<usize, FixedState> {
        &self.adj[node_id as usize]
    }

    /// Returns the set of incident edges refs for a given node.
    pub fn iter_incident_edges(
        &self,
        node_id: NodeId,
    ) -> impl Iterator<Item = (EdgeId, EdgeRef<'_, W>)> {
        self.adj[node_id as usize]
            .iter()
            .map(move |&id| (id as EdgeId, self.csr.get_edge_by_id(id as EdgeId)))
    }
}
