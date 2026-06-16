use foldhash::fast::FixedState;

use crate::{
    graph::{
        AdjList, HypergraphAccessor, NodeId, NodeWeight, UnweightedHypergraph, WeightedHypergraph,
    },
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

const TRIANGLE: CompactMotif<3> = {
    let mut rv = CompactMotif::<3>::zero();
    rv.const_add_edge_with_nodes(CompressedNodeSet::from_array([0, 1]));
    rv.const_add_edge_with_nodes(CompressedNodeSet::from_array([0, 2]));
    rv.const_add_edge_with_nodes(CompressedNodeSet::from_array([1, 2]));
    rv
};

const STRAIGHT_PATH: CompactMotif<3> = {
    let mut rv = CompactMotif::<3>::zero();
    rv.const_add_edge_with_nodes(CompressedNodeSet::from_array([0, 1]));
    rv.const_add_edge_with_nodes(CompressedNodeSet::from_array([1, 2]));
    rv
};

pub fn unweighted_3(hg: &UnweightedHypergraph) -> HashMap<Fingerprint3, MotifStats> {
    let hg = &hg.0;
    let mut motif_stats = HashMap::new();
    let mut triangles = MotifStats::new();
    let mut straight_paths = MotifStats::new();

    let edges_2: Vec<(NodeId, NodeId, ())> = hg
        .edges::<2>()
        .iter()
        .cloned()
        .map(|e| (e.nodes[0], e.nodes[1], ()))
        .collect();

    let (mut adj, direct_map, inverse_map) = AdjList::from_edges_mapped(edges_2);
    adj.make_undirected();

    let mut adj_hash: Vec<HashMap<NodeId, (), FixedState>> = adj
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

    let triangle_fingeprint = TRIANGLE.fingerprint();
    let straight_path_fingerprint = STRAIGHT_PATH.fingerprint();

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
            && adj_hash[*a as usize].contains_key(b)
        {
            inner_edges[inner_count] = CompressedNodeSet::from_array([0, 1]);
            inner_count += 1;
        }
        if let (Some(a), Some(c)) = (a, c)
            && adj_hash[*a as usize].contains_key(c)
        {
            inner_edges[inner_count] = CompressedNodeSet::from_array([0, 2]);
            inner_count += 1;
        }
        if let (Some(b), Some(c)) = (b, c)
            && adj_hash[*b as usize].contains_key(c)
        {
            inner_edges[inner_count] = CompressedNodeSet::from_array([1, 2]);
            inner_count += 1;
        }

        let motif = {
            let mut rv = CompactMotif::<3>::zero();
            rv.add_edge_with_nodes(CompressedNodeSet::from_array([0, 1, 2]));
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
    let hg = &hg.0;
    let mut motif_stats = HashMap::new();
    let mut triangles = MotifStats::new();
    let mut straight_paths = MotifStats::new();

    let edges_2: Vec<(NodeId, NodeId, NodeWeight)> = hg
        .edges::<2>()
        .iter()
        .cloned()
        .map(|e| (e.nodes[0], e.nodes[1], e.weight))
        .collect();

    let (mut adj, direct_map, inverse_map) = AdjList::from_edges_mapped(edges_2);
    adj.make_undirected();

    let mut adj_hash: Vec<HashMap<NodeId, NodeWeight, FixedState>> = adj
        .adj
        .iter()
        .cloned()
        .map(|neighboors| neighboors.into_iter().collect())
        .collect();

    // Computing the sum of products of all possible 3 pairs of incident pairs for each vertex using Newtoon sum
    // O(n + e)
    let mut s1: Vec<f64> = vec![0.0; adj.adj.len()];
    let mut s2: Vec<f64> = vec![0.0; adj.adj.len()];
    // let mut s3: Vec<f64> = vec![0.0; adj.adj.len()];

    for edge in hg.edges::<2>().iter() {
        s1[inverse_map[&edge.nodes[0]] as usize] += edge.weight.sqrt() as f64;
        s1[inverse_map[&edge.nodes[1]] as usize] += edge.weight.sqrt() as f64;

        s2[inverse_map[&edge.nodes[0]] as usize] += edge.weight as f64;
        s2[inverse_map[&edge.nodes[1]] as usize] += edge.weight as f64;
    }

    straight_paths.mean_intensity = {
        let mut tot_2_intensity = 0.0;
        for i in 0..adj.adj.len() {
            tot_2_intensity += (s1[i] * s1[i] - s2[i]) as f64 / 2.0;
        }
        tot_2_intensity
    };

    straight_paths.count = {
        let mut tot_2_count = 0;
        for neighboors in &adj.adj {
            tot_2_count += neighboors.len() * (neighboors.len() - 1) / 2;
        }
        tot_2_count
    };

    forward_hashed_cloj(&adj, false, |a, b, c| {
        let w_ab = *adj_hash[a as usize].get(&b).unwrap() as f64;
        let w_ac = *adj_hash[a as usize].get(&c).unwrap() as f64;
        let w_bc = *adj_hash[b as usize].get(&c).unwrap() as f64;

        triangles.count += 1;
        straight_paths.count -= 3;
        triangles.mean_intensity += (w_ab * w_ac * w_bc).cbrt();
        straight_paths.mean_intensity -=
            (w_ab * w_ac).sqrt() + (w_ab * w_bc).sqrt() + (w_ac * w_bc).sqrt();
    });

    let triangle_fingeprint = TRIANGLE.fingerprint();
    let straight_path_fingerprint = STRAIGHT_PATH.fingerprint();

    // 3-edge counting
    for edge_e in hg.edges::<3>().iter() {
        let (a, b, c) = (
            inverse_map.get(&edge_e.nodes[0]),
            inverse_map.get(&edge_e.nodes[1]),
            inverse_map.get(&edge_e.nodes[2]),
        );
        let mut inner_count = 0;
        let mut inner_inensity = 1.0;
        let mut inner_edges = [CompressedNodeSet::new(0); 3];
        let mut inner_weights = [0.0; 3];

        if let (Some(a), Some(b)) = (a, b)
            && adj_hash[*a as usize].contains_key(b)
        {
            inner_edges[inner_count] = CompressedNodeSet::from_array([0, 1]);
            inner_weights[inner_count] = *adj_hash[*a as usize].get(b).unwrap() as f64;

            inner_inensity *= inner_weights[inner_count];
            inner_count += 1;
        }
        if let (Some(a), Some(c)) = (a, c)
            && adj_hash[*a as usize].contains_key(c)
        {
            inner_edges[inner_count] = CompressedNodeSet::from_array([0, 2]);
            inner_weights[inner_count] = *adj_hash[*a as usize].get(c).unwrap() as f64;

            inner_inensity *= inner_weights[inner_count];
            inner_count += 1;
        }
        if let (Some(b), Some(c)) = (b, c)
            && adj_hash[*b as usize].contains_key(c)
        {
            inner_edges[inner_count] = CompressedNodeSet::from_array([1, 2]);
            inner_weights[inner_count] = *adj_hash[*b as usize].get(c).unwrap() as f64;

            inner_inensity *= inner_weights[inner_count];
            inner_count += 1;
        }

        let motif = {
            let mut rv = CompactMotif::<3>::zero();
            rv.add_edge_with_nodes(CompressedNodeSet::from_array([0, 1, 2]));
            for i in 0..inner_count {
                rv.add_edge_with_nodes(inner_edges[i]);
            }
            rv
        };

        let curr_stats = motif_stats
            .entry(motif.fingerprint())
            .or_insert_with(MotifStats::new);
        curr_stats.count += 1;
        curr_stats.mean_intensity +=
            (inner_inensity * edge_e.weight as f64).powf(1.0 / (inner_count as f64 + 1.0));

        if inner_count == 2 {
            straight_paths.count -= 1;
            straight_paths.mean_intensity -= (inner_weights[0] * inner_weights[1]).sqrt();
        }
        if inner_count == 3 {
            triangles.mean_intensity -=
                (inner_weights[0] * inner_weights[1] * inner_weights[2]).cbrt();
            triangles.count -= 1;
        }
    }

    motif_stats.insert(triangle_fingeprint, triangles);
    motif_stats.insert(straight_path_fingerprint, straight_paths);

    for (_, stats) in motif_stats.iter_mut() {
        stats.mean_intensity /= stats.count as f64;
    }

    motif_stats
}

pub fn unweighted_4(hg: &UnweightedHypergraph) {
    let hg = &hg.0; // this is a &Hypergraph<u32, ()>
}

// let mut motif_stats = HashMap::new();
// let mut triangles = MotifStats::new();
// let mut straight_paths = MotifStats::new();
//
// let edges_2: Vec<(NodeId, NodeId, ())> = hg
//     .edges::<2>()
//     .iter()
//     .cloned()
//     .map(|e| (e.nodes[0], e.nodes[1], ()))
//     .collect();
//
// let (mut adj, direct_map, inverse_map) = AdjList::from_edges_mapped(edges_2);
// adj.make_undirected();
//
// let mut adj_hash: Vec<HashMap<NodeId, (), FixedState>> = adj
//     .adj
//     .iter()
//     .cloned()
//     .map(|neighboors| neighboors.into_iter().collect())
//     .collect();
//
// forward_hashed_cloj(&adj, false, |a, b, c| {
//     triangles.count += 1;
// });

pub fn weighted_4(hg: &WeightedHypergraph) {}

pub fn unweighted_5(hg: &UnweightedHypergraph) {}

pub fn weighted_5(hg: &WeightedHypergraph) {}
