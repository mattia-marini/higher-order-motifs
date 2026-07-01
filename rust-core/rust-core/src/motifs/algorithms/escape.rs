use foldhash::fast::FixedState;
use hashbrown::HashMap;

use crate::{
    graph::{AdjList, Hypergraph, NodeId, NodeWeight},
    misc::{
        common_neighbors_sorted_list_3_cloj,
        cycle::{count_c4, count_c4_no_sort},
        degree_ordering, sort_by_degree,
    },
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
    let (order, rank, max_deg) = sort_by_degree(&mut adj, false);

    // After degree sorting, for each node u, the number of neighbors v such that v ≺ u
    let n_less_count = adj
        .adj
        .iter()
        .enumerate()
        .map(|(v, neighbors)| {
            neighbors
                .iter()
                .filter(|(&&(neighbor, ref weight))| {
                    adj.adj[neighbor as usize].len() < neighbors.len()
                        || (adj.adj[neighbor as usize].len() == neighbors.len()
                            && (neighbor as usize) < v)
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
    let mut adj_hash: Vec<HashMap<NodeId, NodeId, FixedState>> = adj
        .adj
        .iter()
        .cloned()
        .map(|neighbors| neighbors.into_iter().map(|e| (e.0, e.1.0)).collect())
        .collect();

    // Initialize motif stats
    let mut triangle = MotifStats::new();

    let mut path3 = MotifStats::new();
    let mut star3 = MotifStats::new();

    let mut k4 = MotifStats::new();
    let mut c4 = MotifStats::new();

    let mut diamond = MotifStats::new();
    let mut tailed_triangle = MotifStats::new();

    let mut tri_edge = vec![0; hg.m()];
    let mut tri_vertex = vec![0; hg.n()];

    // Count triangles with forward hashed in O(m^1.5)
    // let mut triangles = Vec::new();

    // Compute triangles + cliques
    // TODO: make it use degeneracy order instead of degree order
    forward_hashed_cloj(&adj, false, |a, b, c| {
        let upper_bound = a.min(b).min(c);
        let edge_ab = adj_hash[a as usize][&b] as usize;
        let edge_ac = adj_hash[a as usize][&c] as usize;
        let edge_bc = adj_hash[b as usize][&c] as usize;

        triangle.count += 1;

        tri_edge[edge_ab] += 1;
        tri_edge[edge_ac] += 1;
        tri_edge[edge_bc] += 1;

        tri_vertex[a as usize] += 1;
        tri_vertex[b as usize] += 1;
        tri_vertex[c as usize] += 1;
        // 4-clique counting
        common_neighbors_sorted_list_3_cloj(&adj[a], &adj[b], &adj[c], upper_bound, |i, j, k| {
            let common = adj[a][i].0;
            k4.count += 1;
            // Add K4
        });
    });

    // Compute 4-cycles
    c4.count = count_c4_no_sort(&adj, &order);

    // Compute other non-induced counts
    for x in 0..adj.n() {
        let deg_x = adj.adj[x].len();
        star3.count += deg_x * (deg_x - 1) * (deg_x - 2) / 6;
        tailed_triangle.count += tri_vertex[x] * (deg_x - 2);

        for y in 0..n_less_count[x] {
            let neighbor_y = adj.adj[x][y].0 as usize;
            let edge_xy = adj.adj[x][y].1.0 as usize;
            let deg_y = adj.adj[neighbor_y].len();

            path3.count += (deg_x - 1) * (deg_y - 1);
            diamond.count += tri_edge[edge_xy] * (tri_edge[edge_xy] - 1) / 2;
        }
    }
    path3.count -= 3 * triangle.count;

    todo!()
}

pub fn weighted_4(hg: &Hypergraph<NodeId, NodeWeight>) -> HashMap<Fingerprint4, MotifStats> {
    todo!()
}
