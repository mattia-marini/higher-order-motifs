use std::ops::BitOr;

use foldhash::fast::FixedState;
use hashbrown::{HashMap, HashSet};

use crate::{
    misc::{
        common_neighbors_sorted_list_3_by_key, common_neighbors_sorted_list_3_cloj,
        cycle::{count_c4, count_c4_no_sort},
        degeneracy_ordering, degree_ordering, sort_by_degree,
    },
    motifs::{
        algorithms::const_graphlets::{
            DIAMOND, FOUR_CLIQUE, FOUR_CYCLE, PATH_4, STAR_4, TAILED_TRIANGLE,
        },
        compressed_motif::CompactMotif,
        compressed_node_set::CompressedNodeSet,
        fingerprint::Fingerprint4,
        types::MotifStats,
    },
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

    let mut path4 = MotifStats::new();
    let mut star4 = MotifStats::new();

    let mut k4 = MotifStats::new();
    let mut c4 = MotifStats::new();

    let mut diamond = MotifStats::new();
    let mut tailed_triangle = MotifStats::new();

    let mut tri_edge = vec![0; adj.m()];
    let mut tri_vertex = vec![0; adj.n()];

    adj_list.sort_neighbors();
    let (order, pos, degeneracy) = degeneracy_ordering(&adj_list);
    // println!("Adj list: {:?}", adj_list);
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

        // let na = adj_list[a].iter().map(|e| e.node).collect::<Vec<_>>();
        // let nb = adj_list[b].iter().map(|e| e.node).collect::<Vec<_>>();
        // let nc = adj_list[c].iter().map(|e| e.node).collect::<Vec<_>>();

        // 4-clique counting
        common_neighbors_sorted_list_3_by_key(
            &adj_list[a],
            &adj_list[b],
            &adj_list[c],
            &upper_bound,
            |e| &e.node,
            |i, j, k| {
                let common = adj_list[a][i].node;
                k4.count += 1;
                // Add K4
            },
        );
    });

    // Compute other non-induced counts
    for x in 0..adj_list.n() {
        let deg_x = adj_list[x].len();
        star4.count += deg_x * (deg_x - 1) * (deg_x - 2) / 6;
        tailed_triangle.count += tri_vertex[x] * (deg_x - 2);

        let mut y = 0;
        loop {
            if y >= adj_list[x].len() {
                break;
            }
            let neighbor_y = adj_list[x][y].node as usize;
            if neighbor_y >= x {
                break;
            }

            let neighbor_y = adj_list[x][y].node as usize;
            let edge_xy = adj_list[x][y].edge as usize;
            let deg_y = adj_list[neighbor_y].len();

            path4.count += (deg_x - 1) * (deg_y - 1);
            diamond.count += tri_edge[edge_xy] * (tri_edge[edge_xy] - 1) / 2;

            y += 1;
        }
    }
    path4.count -= 3 * triangle.count;

    // c4 are enumerated efficiently
    c4.count = count_c4(&mut adj_list);

    // println!("adj list: n {}, m {}", adj_list.n(), adj_list.m());
    // println!("Tot triangles: {}", triangle.count);
    // println!("Tot K4: {}", k4.count);
    // println!("Tot C4: {}", c4.count);

    // println!("{:?}", adj_list);
    // After degree sorting, for each node u, the number of neighbors v such that v ≺ u
    // let (order, rank, max_deg) = sort_by_degree(&mut adj_list, false);
    // let n_less_count = adj_list
    //     .iter_neighbors()
    //     .enumerate()
    //     .map(|(v, neighbors)| {
    //         neighbors
    //             .iter()
    //             // .filter(|(&&(neighbor, ref weight))| {
    //             .filter(|n| {
    //                 adj_list[n.node].len() < neighbors.len()
    //                     || (adj_list[n.node].len() == neighbors.len() && n.node < v as NodeId)
    //             })
    //             .count()
    //     })
    //     .collect::<Vec<usize>>();

    // converting to induced counts
    diamond.count -= 6 * k4.count;
    c4.count -= 3 * k4.count + diamond.count;
    tailed_triangle.count -= 12 * k4.count + 4 * diamond.count;
    star4.count -= 4 * k4.count + 2 * diamond.count + tailed_triangle.count;
    path4.count -= 12 * k4.count + 6 * diamond.count + 2 * tailed_triangle.count + 4 * c4.count;

    let mut rv = HashMap::new();

    println!(
        "path4: {}, star4: {}, k4: {}, c4: {}, diamond: {}, tailed_triangle: {}",
        path4.count, star4.count, k4.count, c4.count, diamond.count, tailed_triangle.count
    );

    // Add results to the motif stats hashmap
    rv.insert(PATH_4.fingerprint(), path4);
    rv.insert(STAR_4.fingerprint(), star4);

    rv.insert(FOUR_CLIQUE.fingerprint(), c4);
    rv.insert(FOUR_CYCLE.fingerprint(), k4);

    rv.insert(DIAMOND.fingerprint(), diamond);
    rv.insert(TAILED_TRIANGLE.fingerprint(), tailed_triangle);

    return rv;
    // Hyper degeneracy to efficiently find inclusions of every edge

    for (pivot_edge_id, pivot_edge) in adj.iter_by_size(3) {
        let nodes = pivot_edge.nodes;
        let min_inner_node = *nodes.iter().min().unwrap();

        let mut mapped_nodes = [0; 3];
        mapped_nodes[nodes[0] as usize] = 0;
        mapped_nodes[nodes[1] as usize] = 1;
        mapped_nodes[nodes[2] as usize] = 2;

        let nodes_set = {
            let mut rv = HashSet::new();
            for n in nodes.iter() {
                rv.insert(*n);
            }
            rv
        };

        let mut center_motif = const {
            let mut motif_3 = CompactMotif::<4>::zero();
            motif_3.const_add_edge_with_nodes(CompressedNodeSet::from_array([0, 1, 2]));

            let motif_2 = CompactMotif::<4>::zero();
            [motif_2, motif_3] // keeping them separated to subtract overcounted 2-uniform-motifs fast
        };

        let mut extension_nodes = HashMap::new();

        for i in 0..3 {
            for (edge_id, edge) in adj.iter_incident_edges(nodes[i]) {
                if edge.nodes.len() < 4 && pivot_edge_id != edge_id {
                    let (mut i1, mut i2) = (0, 0);
                    let mut inner_nodes = [0; 2];
                    let mut outer_nodes = [0; 2];

                    for n in edge.nodes {
                        if nodes_set.contains(n) {
                            inner_nodes[i1] = *n;
                            i1 += 1;
                        } else {
                            outer_nodes[i2] = *n;
                            i2 += 1;
                        }
                    }

                    if i2 == 0 {
                        center_motif[0].add_edge_with_nodes(CompressedNodeSet::from_iter(
                            inner_nodes[0..i1].into_iter().map(|e| *e as u8),
                        ));
                    }

                    if outer_nodes.len() == 1 {
                        //avoid counting over counting
                        if edge.nodes.len() == 3 && !(min_inner_node == outer_nodes[0]) {
                            continue;
                        }

                        let mut nodes = CompressedNodeSet::new(0);

                        for n in inner_nodes {
                            nodes.insert(mapped_nodes[n as usize]);
                        }

                        // assuming the added node is the last one without loss of generality
                        nodes.insert(3);

                        let motif = {
                            let mut rv = CompactMotif::<4>::zero();
                            rv.add_edge_with_nodes(nodes);
                            rv
                        };

                        let mut peripheral_motifs = [CompactMotif::<4>::zero(); 2];
                        let bucket = edge.nodes.len() - 2;
                        peripheral_motifs[bucket] = motif;

                        extension_nodes
                            .entry(outer_nodes[0])
                            .and_modify(|motifs: &mut [CompactMotif<4>; 2]| {
                                motifs[bucket] = motifs[bucket].bitor(motif)
                            })
                            .or_insert(peripheral_motifs);
                    }
                }
            }
        }

        for (node, motifs) in extension_nodes {
            let uniform_2_motf = center_motif[0].bitor(motifs[0]);
            let uniform_3_motf = center_motif[1].bitor(motifs[1]);
            let combined = uniform_2_motf.bitor(uniform_3_motf);

            // Correcting overcounting of 2-uniform motifs
            rv.entry(uniform_2_motf.fingerprint())
                .and_modify(|stats: &mut MotifStats| stats.count -= 1);

            rv.entry(combined.fingerprint())
                .and_modify(|stats: &mut MotifStats| stats.count += 1)
                .or_insert(MotifStats::new());
        }
    }

    rv
}

pub fn weighted_4(hg: &HyperAdjList<NodeWeight>) -> HashMap<Fingerprint4, MotifStats> {
    todo!()
}
