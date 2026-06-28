use foldhash::fast::FixedState;
use hashbrown::HashMap;

use crate::{
    graph::{AdjList, Hypergraph, NodeId, NodeWeight},
    misc::common_neighbors_sorted_list_3_cloj,
    motifs::{fingerprint::Fingerprint4, types::MotifStats},
    triangle::forward::forward_hashed_cloj,
};

pub fn unweighted_4(hg: &Hypergraph<NodeId, ()>) -> HashMap<Fingerprint4, MotifStats> {
    // let mut motif_stats = HashMap::new();

    // Extract 2-edges (regular edges) and build adjacency list
    let edges_2: Vec<(NodeId, NodeId, ())> = hg
        .edges::<2>()
        .iter()
        .cloned()
        .map(|e| (e.nodes[0], e.nodes[1], ()))
        .collect();

    let (mut adj, _direct_map, inverse_map) = AdjList::incidence_from_edges_mapped(edges_2);
    adj.make_undirected();
    adj.sort_neighbors();

    // let mut inc_list = adj.clone();
    // for u in 0..adj.n() {
    //     for v in inc_list[u] {}
    // }

    // Create hash-based adjacency for O(1) edge lookups
    // adj_hash[i] = HashMap<(neighbor_node_id, edge_id)>
    let mut adj_hash: Vec<HashMap<NodeId, NodeId, FixedState>> = adj
        .adj
        .iter()
        .cloned()
        .map(|neighbors| neighbors.into_iter().map(|e| (e.0, e.1.0)).collect())
        .collect();

    // Initialize motif stats
    let mut four_clique = MotifStats::new();
    let mut diamond = MotifStats::new();
    let mut four_cycle = MotifStats::new();
    let mut paw = MotifStats::new();
    let mut path_4 = MotifStats::new();
    let mut star_4 = MotifStats::new();
    let mut two_edges_disconnected = MotifStats::new();
    let mut triangle_plus_isolated = MotifStats::new();
    let mut path_3_plus_isolated = MotifStats::new();
    let mut edge_plus_two_isolated = MotifStats::new();
    let mut four_isolated = MotifStats::new();

    let mut tri = vec![0; hg.m()];

    // Count triangles with forward hashed in O(m^1.5)
    let mut triangles = Vec::new();

    // TODO: make it use degeneracy order instead of degree order
    forward_hashed_cloj(&adj, false, |a, b, c| {
        triangles.push((a, b, c));
        let upper_bound = a.min(b).min(c);
        let edge_ab = adj_hash[a as usize][&b] as usize;
        let edge_ac = adj_hash[a as usize][&c] as usize;
        let edge_bc = adj_hash[b as usize][&c] as usize;

        tri[edge_ab] += 1;
        tri[edge_ac] += 1;
        tri[edge_bc] += 1;

        // 4-clique counting
        common_neighbors_sorted_list_3_cloj(&adj[a], &adj[b], &adj[c], upper_bound, |i, j, k| {
            let common = adj[a][i].0;

            // Add K4
        });
    });
    todo!()
}

pub fn weighted_4(hg: &Hypergraph<NodeId, NodeWeight>) -> HashMap<Fingerprint4, MotifStats> {
    todo!()
}
