use foldhash::fast::FixedState;

use crate::{
    graph::{AdjList, HypergraphAccessor, NodeId, UnweightedHypergraph, WeightedHypergraph},
    motifs::{
        compressed_motif::{CompactMotif, CompactMotif3},
        compressed_node_set::CompressedNodeSet,
        fingerprint::Fingerprint3,
        types::MotifStats,
    },
};

use crate::triangle::forward::forward_hashed_cloj;

use hashbrown::{HashMap, HashSet};
use pyo3::pyfunction;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

pub fn unweighted_3(hg: &UnweightedHypergraph) -> HashMap<Fingerprint3, MotifStats> {
    let hg = &hg.0;
    let mut motif_stats = HashMap::new();
    let mut triangles = MotifStats::new();
    let mut straight_paths = MotifStats::new();

    let edges_2: Vec<(NodeId, NodeId)> = hg
        .edges::<2>()
        .iter()
        .cloned()
        .map(|e| (e.nodes[0], e.nodes[1]))
        .collect();

    let (mut adj, direct_map, inverse_map) = AdjList::from_edges_mapped(edges_2);
    adj.make_undirected();

    let mut adj_hash: Vec<HashSet<NodeId, FixedState>> = adj
        .adj
        .iter()
        .cloned()
        .map(|neighboors| neighboors.into_iter().collect())
        .collect();

    forward_hashed_cloj(&adj, false, |a, b, c| {
        triangles.count += 1;
    });

    let mut tot_2_edges_motifs_count = 0;
    for neighboors in &adj.adj {
        tot_2_edges_motifs_count += neighboors.len() * (neighboors.len() - 1) / 2;
    }
    straight_paths.count = tot_2_edges_motifs_count - 3 * triangles.count;

    let triangle_rep = {
        let mut rv = CompactMotif::<3>::zero();
        rv.add_edge_with_nodes(CompressedNodeSet::from_nodes([0, 1]));
        rv.add_edge_with_nodes(CompressedNodeSet::from_nodes([0, 2]));
        rv.add_edge_with_nodes(CompressedNodeSet::from_nodes([1, 2]));
        rv
    };
    let triangle_fingeprint = triangle_rep.fingerprint();

    let straight_path = {
        let mut rv = CompactMotif::<3>::zero();
        rv.add_edge_with_nodes(CompressedNodeSet::from_nodes([0, 1]));
        rv.add_edge_with_nodes(CompressedNodeSet::from_nodes([1, 2]));
        rv
    };
    let straight_path_fingerprint = straight_path.fingerprint();

    motif_stats.insert(triangle_fingeprint, triangles);
    motif_stats.insert(straight_path_fingerprint, straight_paths);

    for edge_e in hg.edges::<3>().iter() {
        let (a, b, c) = (
            inverse_map.get(&edge_e.nodes[0]),
            inverse_map.get(&edge_e.nodes[1]),
            inverse_map.get(&edge_e.nodes[2]),
        );
        let mut inner_count = 0;
        let mut inner_edges = [CompressedNodeSet::new(0); 3];

        if let (Some(a), Some(b)) = (a, b)
            && adj_hash[*a as usize].contains(b)
        {
            inner_edges[inner_count] = CompressedNodeSet::from_nodes([0, 1]);
            inner_count += 1;
        }
        if let (Some(a), Some(c)) = (a, c)
            && adj_hash[*a as usize].contains(c)
        {
            inner_edges[inner_count] = CompressedNodeSet::from_nodes([0, 2]);
            inner_count += 1;
        }
        if let (Some(b), Some(c)) = (b, c)
            && adj_hash[*b as usize].contains(c)
        {
            inner_edges[inner_count] = CompressedNodeSet::from_nodes([1, 2]);
            inner_count += 1;
        }

        let motif = {
            let mut rv = CompactMotif::<3>::zero();
            rv.add_edge_with_nodes(CompressedNodeSet::from_nodes([0, 1, 2]));
            for i in 0..inner_count {
                rv.add_edge_with_nodes(inner_edges[i]);
            }
            rv
        };

        motif_stats
            .entry(motif.fingerprint())
            .or_insert_with(MotifStats::new)
            .count += 1;

        if inner_count == 2 {
            motif_stats
                .get_mut(&straight_path_fingerprint)
                .unwrap()
                .count -= 1;
        }
        if inner_count == 3 {
            motif_stats.get_mut(&triangle_fingeprint).unwrap().count -= 1;
        }
    }

    motif_stats
}

pub fn weighted_3(hg: &WeightedHypergraph) -> HashMap<Fingerprint3, MotifStats> {
    todo!()
}

pub fn unweighted_4(hg: &UnweightedHypergraph) {}

pub fn weighted_4(hg: &WeightedHypergraph) {}

pub fn unweighted_5(hg: &UnweightedHypergraph) {}

pub fn weighted_5(hg: &WeightedHypergraph) {}
