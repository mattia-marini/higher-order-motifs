use std::ops::BitOr;

use bit_set::BitSet;
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
            DIAMOND, FOUR_CLIQUE, FOUR_CYCLE, PATH_4, STAR_4, STRAIGHT_PATH, TAILED_TRIANGLE,
            TRIANGLE,
        },
        compressed_motif::CompactMotif,
        compressed_node_set::CompressedNodeSet,
        fingerprint::{Fingerprint3, Fingerprint4},
        types::MotifStats,
    },
    triangle::forward::forward_hashed_cloj,
    types::{
        EdgeId, Hypergraph, NodeId, NodeWeight,
        adj_list::{
            AdjList, AdjSet, Undirected, WithIncidence, WithoutIncidence, common::Neighbor,
        },
        hyperadj_list::{HyperAdjList, HyperAdjListBase},
    },
};

pub fn unweighted_3(hg: &Hypergraph<NodeId, ()>) -> HashMap<Fingerprint3, MotifStats> {
    let mut motif_stats = HashMap::new();
    let mut triangles = MotifStats::new();
    let mut straight_paths = MotifStats::new();

    let edges_2: Vec<(NodeId, NodeId, ())> = hg
        .edges::<2>()
        .iter()
        .cloned()
        .map(|e| (e.nodes[0], e.nodes[1], ()))
        .collect();

    // Undirected by design
    let (mut adj, direct_map, inverse_map) =
        AdjList::<(), Undirected, WithIncidence>::from_edges_mapped(edges_2);
    let adj_hash: AdjSet<(), Undirected, WithoutIncidence> = adj.clone().into();

    // adj.make_undirected();

    // let mut adj_hash: Vec<HashMap<NodeId, (), FixedState>> = adj
    //     .adj
    //     .iter()
    //     .cloned()
    //     .map(|neighboors| neighboors.into_iter().collect())
    //     .collect();

    let (order, pos, degeneracy) = degeneracy_ordering(&adj);
    forward_hashed_cloj(&adj, Some((&order, &pos)), |a, b, c| {
        triangles.count += 1;
    });

    let mut tot_2_edges_motifs_count = 0;
    for neighboors in adj.iter_neighbors() {
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

pub fn weighted_3(hg: &Hypergraph<NodeId, NodeWeight>) -> HashMap<Fingerprint3, MotifStats> {
    let mut motif_stats = HashMap::new();
    let mut triangles = MotifStats::new();
    let mut straight_paths = MotifStats::new();

    let edges_2: Vec<(NodeId, NodeId, NodeWeight)> = hg
        .edges::<2>()
        .iter()
        .cloned()
        .map(|e| (e.nodes[0], e.nodes[1], e.weight))
        .collect();

    let (mut adj, direct_map, inverse_map) =
        AdjList::<NodeWeight, Undirected, WithIncidence>::from_edges_mapped(edges_2);
    let adj_hash: AdjSet<NodeWeight, Undirected, WithoutIncidence> = adj.clone().into();

    // Computing the sum of products of all possible 3 pairs of incident pairs for each vertex using
    // Newtoon sum O(n + e)
    let mut s1: Vec<f64> = vec![0.0; adj.n()];
    let mut s2: Vec<f64> = vec![0.0; adj.n()];
    // let mut s3: Vec<f64> = vec![0.0; adj.adj.len()];

    for edge in hg.edges::<2>().iter() {
        s1[inverse_map[&edge.nodes[0]] as usize] += edge.weight.sqrt() as f64;
        s1[inverse_map[&edge.nodes[1]] as usize] += edge.weight.sqrt() as f64;

        s2[inverse_map[&edge.nodes[0]] as usize] += edge.weight as f64;
        s2[inverse_map[&edge.nodes[1]] as usize] += edge.weight as f64;
    }

    straight_paths.mean_intensity = {
        let mut tot_2_intensity = 0.0;
        for i in 0..adj.n() {
            tot_2_intensity += (s1[i] * s1[i] - s2[i]) as f64 / 2.0;
        }
        tot_2_intensity
    };

    straight_paths.count = {
        let mut tot_2_count = 0;
        for neighboors in adj.iter_neighbors() {
            tot_2_count += neighboors.len() * (neighboors.len() - 1) / 2;
        }
        tot_2_count
    };

    let (order, pos, degeneracy) = degeneracy_ordering(&adj);
    forward_hashed_cloj(&adj, Some((&order, &pos)), |a, b, c| {
        let w_ab = adj_hash[a].get(&b).unwrap().0 as f64;
        let w_ac = adj_hash[a].get(&c).unwrap().0 as f64;
        let w_bc = adj_hash[b].get(&c).unwrap().0 as f64;

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
            inner_weights[inner_count] = adj_hash[*a].get(b).unwrap().0 as f64;

            inner_inensity *= inner_weights[inner_count];
            inner_count += 1;
        }
        if let (Some(a), Some(c)) = (a, c)
            && adj_hash[*a as usize].contains_key(c)
        {
            inner_edges[inner_count] = CompressedNodeSet::from_array([0, 2]);
            inner_weights[inner_count] = adj_hash[*a].get(c).unwrap().0 as f64;

            inner_inensity *= inner_weights[inner_count];
            inner_count += 1;
        }
        if let (Some(b), Some(c)) = (b, c)
            && adj_hash[*b as usize].contains_key(c)
        {
            inner_edges[inner_count] = CompressedNodeSet::from_array([1, 2]);
            inner_weights[inner_count] = adj_hash[*b].get(c).unwrap().0 as f64;

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

pub fn unweighted_4(adj: &HyperAdjList<()>) -> HashMap<Fingerprint4, MotifStats> {
    let edges_2: Vec<(NodeId, NodeId, ())> = adj
        .iter_by_size(2)
        .map(|(_, e)| (e.nodes[0], e.nodes[1], ()))
        .collect();

    let (mut adj_list, _direct_map, inverse_map) =
        AdjList::<(), Undirected, WithIncidence>::from_edges_mapped(edges_2);
    let adj_set: AdjSet<(), Undirected, WithIncidence> = adj_list.clone().into();

    let mut rv = HashMap::new();

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

    // converting to induced counts
    diamond.count -= 6 * k4.count;
    c4.count -= 3 * k4.count + diamond.count;
    tailed_triangle.count -= 12 * k4.count + 4 * diamond.count;
    star4.count -= 4 * k4.count + 2 * diamond.count + tailed_triangle.count;
    path4.count -= 12 * k4.count + 6 * diamond.count + 2 * tailed_triangle.count + 4 * c4.count;

    // Add results to the motif stats hashmap
    rv.insert(PATH_4.fingerprint(), path4);
    rv.insert(STAR_4.fingerprint(), star4);

    rv.insert(FOUR_CLIQUE.fingerprint(), k4);
    rv.insert(FOUR_CYCLE.fingerprint(), c4);

    rv.insert(DIAMOND.fingerprint(), diamond);
    rv.insert(TAILED_TRIANGLE.fingerprint(), tailed_triangle);

    let mut mapped_nodes = vec![u8::MAX; adj.n()];
    let mut black_nodes = BitSet::with_capacity(adj.n());
    let mut inserted = BitSet::with_capacity(adj.n());
    let mut extension_nodes = vec![[CompactMotif::<4>::zero(); 2]; adj.n()];
    let mut node_list = Vec::with_capacity(adj.n() / 2);

    // let mut black_nodes = HashSet::new();
    // let mut extension_nodes = HashMap::new();

    for (pivot_edge_id, pivot_edge) in adj.iter_by_size(3) {
        // println!("pivot edge: {:?}", pivot_edge_id);
        let nodes = pivot_edge.nodes;
        let min_inner_node = *nodes.iter().min().unwrap();

        // let mut mapped_nodes = [0; 3];
        mapped_nodes[nodes[0] as usize] = 0;
        mapped_nodes[nodes[1] as usize] = 1;
        mapped_nodes[nodes[2] as usize] = 2;

        let mut center_motif = const {
            let mut motif_3 = CompactMotif::<4>::zero();
            motif_3.const_add_edge_with_nodes(CompressedNodeSet::from_array([0, 1, 2]));

            let motif_2 = CompactMotif::<4>::zero();
            [motif_2, motif_3] // keeping them separated to subtract overcounted 2-uniform-motifs fast
        };

        for i in 0..3 {
            for (edge_id, edge) in adj.iter_incident_edges(nodes[i]) {
                if edge.nodes.len() < 4 && pivot_edge_id != edge_id {
                    let (mut i1, mut i2) = (0, 0);
                    let mut inner_nodes = [0; 2];
                    let mut outer_nodes = [0; 2];

                    for n in edge.nodes {
                        if mapped_nodes[*n as usize] != u8::MAX {
                            inner_nodes[i1] = *n;
                            i1 += 1;
                        } else {
                            outer_nodes[i2] = *n;
                            i2 += 1;
                        }
                    }

                    if i2 == 0 {
                        center_motif[0].add_edge_with_nodes(CompressedNodeSet::from_iter(
                            inner_nodes[0..i1]
                                .into_iter()
                                .map(|e| mapped_nodes[*e as usize]),
                        ));
                    } else if i2 == 1 {
                        let outer_node = outer_nodes[0] as usize;
                        //avoid counting over counting
                        if edge.nodes.len() == 3 && !(pivot_edge_id < edge_id) {
                            if !inserted.contains(outer_node) {
                                node_list.push(outer_node);
                                inserted.insert(outer_node);
                            }
                            black_nodes.insert(outer_node);
                            // (outer_nodes[0]);
                            // extension_nodes.remove(&outer_nodes[0]);
                            continue;
                        }

                        if black_nodes.contains(outer_node) {
                            continue;
                        }

                        let nodes = {
                            let mut rv = CompressedNodeSet::new(0);
                            for n in inner_nodes[0..i1].iter() {
                                rv.insert(mapped_nodes[*n as usize] as usize);
                            }
                            // assuming the added node is the last one without loss of generality
                            rv.insert(3);
                            rv
                        };

                        let motif = {
                            let mut rv = CompactMotif::<4>::zero();
                            // println!("nodes: {:?}", nodes);
                            rv.add_edge_with_nodes(nodes);
                            rv
                        };

                        let bucket = edge.nodes.len() - 2;
                        let motifs = &mut extension_nodes[outer_node];

                        if inserted.contains(outer_node) {
                            motifs[bucket] = motifs[bucket].bitor(motif);
                        } else {
                            let mut peripheral_motifs = [CompactMotif::<4>::zero(); 2];
                            peripheral_motifs[bucket] = motif;
                            *motifs = peripheral_motifs;
                            inserted.insert(outer_node);
                            node_list.push(outer_node);
                        }
                        // extension_nodes
                        //     .entry(outer_nodes[0])
                        //     .and_modify(|motifs: &mut [CompactMotif<4>; 2]| {
                        //         motifs[bucket] = motifs[bucket].bitor(motif);
                        //     })
                        //     .or_insert(peripheral_motifs);
                    }
                }
            }
        }
        // println!("{:?}", node_list);

        // println!("extension nodes: {:?}", extension_nodes);
        for &node in node_list.iter() {
            if !black_nodes.contains(node) {
                let motifs = &extension_nodes[node];
                let uniform_2_motif = center_motif[0].bitor(motifs[0]);
                let uniform_3_motif = center_motif[1].bitor(motifs[1]);
                let combined = uniform_2_motif.bitor(uniform_3_motif);

                // Correcting overcounting of 2-uniform motifs
                rv.entry(uniform_2_motif.fingerprint())
                    .and_modify(|stats: &mut MotifStats| stats.count -= 1);

                rv.entry(combined.fingerprint())
                    .and_modify(|stats: &mut MotifStats| stats.count += 1)
                    .or_insert(MotifStats {
                        count: 1,
                        mean_intensity: 0.,
                        mean_coherence: 0.,
                        actual_intensity: 0.,
                    });
            }

            black_nodes.remove(node);
            inserted.remove(node);
        }

        node_list.clear();

        mapped_nodes[nodes[0] as usize] = u8::MAX;
        mapped_nodes[nodes[1] as usize] = u8::MAX;
        mapped_nodes[nodes[2] as usize] = u8::MAX;
    }

    let mut edges_2 = HashSet::new();
    let mut edges_3 = HashSet::new();

    for (edge_id, edge) in adj.iter_by_size(2) {
        edges_2.insert(edge.nodes);
    }

    for (edge_id, edge) in adj.iter_by_size(3) {
        edges_3.insert(edge.nodes);
    }

    for (edge_id, edge) in adj.iter_by_size(4) {
        // println!("Edge: {:?}", edge.nodes);
        mapped_nodes[edge.nodes[0] as usize] = 0;
        mapped_nodes[edge.nodes[1] as usize] = 1;
        mapped_nodes[edge.nodes[2] as usize] = 2;
        mapped_nodes[edge.nodes[3] as usize] = 3;

        let mut motif = CompactMotif::<4>::zero();
        for i in 0..4 {
            for j in (i + 1)..4 {
                if edges_2.contains([edge.nodes[i], edge.nodes[j]].as_slice()) {
                    motif.add_edge_with_nodes(CompressedNodeSet::from_array([i as u8, j as u8]));
                }
            }
        }

        for i in 0..4 {
            for j in (i + 1)..4 {
                for k in (j + 1)..4 {
                    if edges_3.contains([edge.nodes[i], edge.nodes[j], edge.nodes[k]].as_slice()) {
                        motif.add_edge_with_nodes(CompressedNodeSet::from_array([
                            i as u8, j as u8, k as u8,
                        ]));
                    }
                }
            }
        }

        rv.entry(motif.fingerprint())
            .and_modify(|stats: &mut MotifStats| stats.count -= 1);

        motif.add_edge_with_nodes(CompressedNodeSet::from_array([0, 1, 2, 3]));

        // println!("Motif: {:?}", motif);
        // println!(
        //     "Reconstructed motif: {}",
        //     motif.fingerprint().get_canonical_rep()
        // );

        rv.entry(motif.fingerprint())
            .and_modify(|stats: &mut MotifStats| stats.count += 1)
            .or_insert(MotifStats {
                count: 1,
                mean_intensity: 0.,
                mean_coherence: 0.,
                actual_intensity: 0.,
            });
    }

    // for (_number, (motif, stats)) in rv.iter().enumerate() {
    //     println!("{}\t{}", stats.count, motif.get_canonical_rep());
    // }

    rv
}

pub fn weighted_4(hg: &HyperAdjList<NodeWeight>) -> HashMap<Fingerprint4, MotifStats> {
    todo!()
}
