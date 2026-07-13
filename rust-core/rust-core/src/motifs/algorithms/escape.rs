use foldhash::fast::FixedState;
use hashbrown::HashMap;

use crate::{
    misc::{
        common_neighbors_sorted_list_3_cloj,
        cycle::{count_c4, count_c4_no_sort},
        degeneracy_ordering, degree_ordering, sort_by_degree,
    },
    motifs::{fingerprint::Fingerprint4, types::MotifStats},
    triangle::forward::forward_hashed_cloj,
    types::{
        EdgeId, Hypergraph, NodeId, NodeWeight,
        adj_list::{AdjList, AdjSet, Undirected, WithIncidence, common::Neighbor},
        hyperadj_list::{HyperAdjList, HyperAdjListBase},
    },
};

pub fn unweighted_4(adj: &HyperAdjList<()>) -> HashMap<Fingerprint4, MotifStats> {
    // let mut motif_stats = HashMap::new();

    let edges_2: Vec<(NodeId, NodeId, ())> = adj
        .iter_by_size(2)
        .map(|(_, e)| (e.nodes[0], e.nodes[1], ()))
        .collect();

    let (mut adj_list, _direct_map, inverse_map) =
        AdjList::<(), Undirected, WithIncidence>::from_edges_mapped(edges_2);
    let adj_set: AdjSet<(), Undirected, WithIncidence> = adj_list.clone().into();

    // adj.remove_multiedges();

    let (order, rank, max_deg) = sort_by_degree(&mut adj_list, false);

    // After degree sorting, for each node u, the number of neighbors v such that v ≺ u
    let n_less_count = adj_list
        .iter_neighbors()
        .enumerate()
        .map(|(v, neighbors)| {
            neighbors
                .iter()
                // .filter(|(&&(neighbor, ref weight))| {
                .filter(|n| {
                    adj_list[n.node].len() < neighbors.len()
                        || (adj_list[n.node].len() == neighbors.len() && n.node < v as NodeId)
                })
                .count()
        })
        .collect::<Vec<usize>>();

    // let mut inc_list = adj.clone();
    // for u in 0..adj.n() {
    //     for v in inc_list[u] {}
    // }

    // Create hash-based adjacency for O(1) edge lookups
    // adj_hash[i] = HashMap<(neighbor_node_id, edge_id)>
    // let mut adj_hash: Vec<HashMap<NodeId, NodeId, FixedState>> = adj_list
    //     .adj
    //     .iter()
    //     .cloned()
    //     .map(|neighbors| neighbors.into_iter().map(|e| (e.0, e.1.0)).collect())
    //     .collect();

    // Initialize motif stats
    let mut triangle = MotifStats::new();

    let mut path3 = MotifStats::new();
    let mut star3 = MotifStats::new();

    let mut k4 = MotifStats::new();
    let mut c4 = MotifStats::new();

    let mut diamond = MotifStats::new();
    let mut tailed_triangle = MotifStats::new();

    let mut tri_edge = vec![0; adj.m()];
    let mut tri_vertex = vec![0; adj.n()];

    let (order, pos, degeneracy) = degeneracy_ordering(&adj_list);
    // Compute triangles + cliques
    // Count triangles with forward hashed in O(m^1.5)
    // TODO: make it use degeneracy order instead of degree order
    forward_hashed_cloj(&adj_list, Some((&order, &pos)), |a, b, c| {
        let upper_bound = a.min(b).min(c);
        let edge_ab = adj_set[a][&b].1 as usize;
        let edge_ac = adj_set[a][&c].1 as usize;
        let edge_bc = adj_set[b][&c].1 as usize;

        triangle.count += 1;

        tri_edge[edge_ab] += 1;
        tri_edge[edge_ac] += 1;
        tri_edge[edge_bc] += 1;

        tri_vertex[a as usize] += 1;
        tri_vertex[b as usize] += 1;
        tri_vertex[c as usize] += 1;

        let upper_bound = Neighbor::new(a.min(b).min(c), (), edge_ab as EdgeId);
        // 4-clique counting
        common_neighbors_sorted_list_3_cloj(
            &adj_list[a],
            &adj_list[b],
            &adj_list[c],
            upper_bound,
            |i, j, k| {
                let common = adj_list[a][i].node;
                k4.count += 1;
                // Add K4
            },
        );
    });

    // Compute 4-cycles
    c4.count = count_c4_no_sort(&adj_list, &order);

    // Compute other non-induced counts
    for x in 0..adj_list.n() {
        let deg_x = adj_list[x].len();
        star3.count += deg_x * (deg_x - 1) * (deg_x - 2) / 6;
        tailed_triangle.count += tri_vertex[x] * (deg_x - 2);

        for y in 0..n_less_count[x] {
            let neighbor_y = adj_list[x][y].node as usize;
            let edge_xy = adj_list[x][y].edge as usize;
            let deg_y = adj_list[neighbor_y].len();

            path3.count += (deg_x - 1) * (deg_y - 1);
            diamond.count += tri_edge[edge_xy] * (tri_edge[edge_xy] - 1) / 2;
        }
    }
    path3.count -= 3 * triangle.count;

    // Hyper degeneracy to efficiently find inclusions of every edge

    HashMap::new()
}

pub fn weighted_4(hg: &HyperAdjList<NodeWeight>) -> HashMap<Fingerprint4, MotifStats> {
    todo!()
}
