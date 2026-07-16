use std::{cmp::max, mem::swap, ops::BitOr};

use bit_set::BitSet;
use foldhash::fast::FixedState;
use hashbrown::{HashMap, HashSet};

use crate::{
    misc::{
        OrderAndPos, common_neighbors_sorted_list_3_by_key, common_neighbors_sorted_list_3_cloj,
        cycle::{count_c4, count_c4_no_sort, intensity_c4},
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
    triangle::forward::{forward_hashed_cloj, forward_sorted_cloj},
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

    let (mut order_pos, degeneracy) = degeneracy_ordering(&adj);
    order_pos.reverse();

    forward_hashed_cloj(&adj, Some(&order_pos), |a, b, c| {
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
    // Newton sum O(n + e)
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

    let (mut order_pos, degeneracy) = degeneracy_ordering(&adj);
    order_pos.reverse();

    forward_hashed_cloj(&adj, Some(&order_pos), |a, b, c| {
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
    let (mut order_pos, degeneracy) = degeneracy_ordering(&adj_list);
    order_pos.reverse();
    // Compute triangles + cliques
    // Count triangles with forward hashed in O(m^1.5)
    // TODO: make it use degeneracy order instead of degree order
    forward_hashed_cloj(&adj_list, Some(&order_pos), |a, b, c| {
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

#[derive(Debug, Clone, Copy)]
struct NewtonPair {
    /// sum of elements
    s1: f32,

    /// sum of squared elements
    s2: f32,
}

impl NewtonPair {
    /// Returns the sum of products of all possible pairs of elements in the set.
    pub fn get_sum_of_products(&self) -> f32 {
        (self.s1 * self.s1 - self.s2) / 2.
    }

    pub fn empty() -> Self {
        NewtonPair { s1: 0., s2: 0. }
    }
}

#[derive(Debug, Clone, Copy)]
struct NewtonTriplet {
    /// sum of elements
    s1: f32,

    /// sum of squared elements
    s2: f32,

    /// sum of cube elements
    s3: f32,
}

impl NewtonTriplet {
    /// Returns the sum of products of all possible pairs of elements in the set.
    pub fn get_sum_of_products(&self) -> f32 {
        (self.s1 * self.s1 * self.s1 - 3.0 * self.s1 * self.s2 + 2.0 * self.s3) / 6.0
    }

    pub fn empty() -> Self {
        NewtonTriplet {
            s1: 0.,
            s2: 0.,
            s3: 0.,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NewtonEdge {
    /// For each triangle incident to the edge, w(a,b)w(a,c)w(b,c)^(1/4)
    s_4: NewtonPair,
    /// For each triangle incident to the edge, w(a,b)w(a,c)w(b,c)^(1/5)
    s_5: NewtonPair,

    /// For each triangle incident to the edge (a, b), w(a,d)^(1/3)
    s3_upper: NewtonPair,
    /// For each triangle incident to the edge (a, b), w(b,d)^(1/3)
    s3_lower: NewtonPair,
}

impl NewtonEdge {
    pub fn empty() -> Self {
        Self {
            s_4: NewtonPair::empty(),
            s_5: NewtonPair::empty(),
            s3_upper: NewtonPair::empty(),
            s3_lower: NewtonPair::empty(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NewtonVertex {
    s_3: NewtonTriplet,
}

#[derive(Debug, Clone)]
struct MotifStatsPair {
    induced: MotifStats,
    non_induced: MotifStats,
}

impl MotifStatsPair {
    pub fn new() -> Self {
        Self {
            induced: MotifStats::new(),
            non_induced: MotifStats::new(),
        }
    }
}

pub fn weighted_4(adj: &HyperAdjList<NodeWeight>) -> HashMap<Fingerprint4, MotifStats> {
    let edges_2 = adj
        .iter_by_size(2)
        .map(|(_, e)| (e.nodes[0], e.nodes[1], *e.weight))
        .collect::<Vec<_>>();

    let (mut adj_list, _direct_map, _inverse_map) =
        AdjList::<NodeWeight, Undirected, WithIncidence>::from_edges_mapped(edges_2);
    // adj_list.sort_neighbors();
    // let adj_set: AdjSet<NodeWeight, Undirected, WithIncidence> = adj_list.clone().into();

    let mut rv = HashMap::new();

    // Final motif stats
    let mut triangle = MotifStats::new();

    let mut path4 = MotifStats::new();
    let mut star4 = MotifStats::new();

    let mut k4 = MotifStats::new();
    let mut c4 = MotifStats::new();

    let mut diamond = MotifStats::new();
    let mut tailed_triangle = MotifStats::new();

    let mut tri_edge_count = vec![0; adj.m()];
    let mut tri_edge_intensity = vec![0.0; adj.m()];
    let mut tri_vertex = vec![0; adj.n()];
    let mut tri_distal_edge = vec![((0.0, 0.0), (0.0, 0.0)); adj.m()];

    // Saving partial stats to convert rom induced to non induced
    // <motif_a>_in_<motif_b> stores the stats of the non induced occurences of motif_a in motif_b

    // diamon
    let mut diamond_in_k4 = MotifStatsPair::new();

    // c4
    let mut c4_in_k4 = MotifStatsPair::new();
    let mut c4_in_diamond = MotifStatsPair::new();

    // paw
    let mut paw_in_k4 = MotifStatsPair::new();
    let mut paw_in_diamond = MotifStatsPair::new();

    // star4
    let mut star4_in_paw = MotifStatsPair::new();
    let mut star4_in_diamond = MotifStatsPair::new();
    let mut star4_in_k4 = MotifStatsPair::new();

    // path4
    let mut path4_in_paw = MotifStatsPair::new();
    let mut path4_in_c4 = MotifStatsPair::new();
    let mut path4_in_diamond = MotifStatsPair::new();
    let mut path4_in_k4 = MotifStatsPair::new();

    /// Coefficient per edge used for fast combinatorial computation of iintensities
    let mut newton_edge = vec![NewtonEdge::empty(); adj.m()];
    let mut newton_vertex = Vec::with_capacity(adj.n());

    let mut neighbors_weight_sum_s1 = Vec::with_capacity(adj_list.n());
    let mut neighbors_weight_sum_s3 = Vec::with_capacity(adj_list.n());
    let mut neighbors_weight_sum_s4 = Vec::with_capacity(adj_list.n());

    for x in 0..adj_list.n() {
        let mut s1 = 0.0;
        let mut s3 = 0.0;
        let mut s4 = 0.0;
        let mut newton_triplet = NewtonTriplet::empty();

        for y in adj_list[x].iter() {
            s1 += y.weight;
            s3 += y.weight.powf(1.0 / 3.0);
            s4 += y.weight.powf(1.0 / 4.0);

            let root = y.weight.powf(1.0 / 3.0);
            newton_triplet.s1 += root;
            newton_triplet.s2 += root * root;
            newton_triplet.s3 += root * root * root;
        }

        newton_vertex.push(NewtonVertex {
            s_3: newton_triplet,
        });
        neighbors_weight_sum_s1.push(s1);
        neighbors_weight_sum_s3.push(s3);
        neighbors_weight_sum_s4.push(s4);
    }

    let (mut order_pos, degeneracy) = degeneracy_ordering(&adj_list);
    order_pos.reverse();

    // Compute triangles + cliques
    // Count triangles with forward hashed in O(m^1.5)
    forward_sorted_cloj(&mut adj_list, Some(&order_pos), |adj_list, t| {
        let a = t.nodes[0] as usize;
        let b = t.nodes[1] as usize;
        let c = t.nodes[2] as usize;

        let edge_ab = t.edges[0] as usize;
        let edge_ac = t.edges[1] as usize;
        let edge_bc = t.edges[2] as usize;

        let weight_ab = *t.weights[0];
        let weight_ac = *t.weights[1];
        let weight_bc = *t.weights[2];

        let prod = weight_ab * weight_ac * weight_bc;

        triangle.count += 1;
        triangle.mean_intensity += prod.powf(1.0 / 3.0) as f64;

        tri_edge_count[edge_ab] += 1;
        tri_edge_count[edge_ac] += 1;
        tri_edge_count[edge_bc] += 1;

        {
            // for paw counting
            let prod = prod.powf(1.0 / 4.0);
            let weight_ab = weight_ab.powf(1.0 / 4.0);
            let weight_ac = weight_ac.powf(1.0 / 4.0);
            let weight_bc = weight_bc.powf(1.0 / 4.0);

            tri_edge_intensity[edge_ab] += prod;
            tri_edge_intensity[edge_ac] += prod;
            tri_edge_intensity[edge_bc] += prod;

            tri_distal_edge[edge_ab].0.0 += weight_ac;
            tri_distal_edge[edge_ab].0.1 += prod * weight_ac;
            tri_distal_edge[edge_ab].1.0 += weight_bc;
            tri_distal_edge[edge_ab].1.1 += prod * weight_bc;

            tri_distal_edge[edge_ac].0.0 += weight_ab;
            tri_distal_edge[edge_ac].0.1 += prod * weight_ab;
            tri_distal_edge[edge_ac].1.0 += weight_bc;
            tri_distal_edge[edge_ac].1.1 += prod * weight_bc;

            tri_distal_edge[edge_bc].0.0 += weight_ab;
            tri_distal_edge[edge_bc].0.1 += prod * weight_ab;
            tri_distal_edge[edge_bc].1.0 += weight_ac;
            tri_distal_edge[edge_bc].1.1 += prod * weight_ac;

            tailed_triangle.mean_intensity +=
                (prod * (neighbors_weight_sum_s4[a] - weight_ab - weight_ac)) as f64;
            tailed_triangle.mean_intensity +=
                (prod * (neighbors_weight_sum_s4[b] - weight_ab - weight_bc)) as f64;
            tailed_triangle.mean_intensity +=
                (prod * (neighbors_weight_sum_s4[c] - weight_ac - weight_bc)) as f64;
        }

        {
            // for star4 counting
            let prod = prod.powf(1.0 / 3.0);
            let weight_ab = weight_ab.powf(1.0 / 3.0);
            let weight_ac = weight_ac.powf(1.0 / 3.0);
            let weight_bc = weight_bc.powf(1.0 / 3.0);

            star4_in_paw.non_induced.mean_intensity +=
                ((prod / weight_bc) * (neighbors_weight_sum_s3[a] - weight_ab - weight_ac)) as f64;
            star4_in_paw.non_induced.mean_intensity +=
                ((prod / weight_ac) * (neighbors_weight_sum_s3[b] - weight_ab - weight_bc)) as f64;
            star4_in_paw.non_induced.mean_intensity +=
                ((prod / weight_ab) * (neighbors_weight_sum_s3[c] - weight_ac - weight_bc)) as f64;
        }

        let s4_s1 = prod.powf(1.0 / 4.0);
        let s4_s2 = prod.powf(2.0 / 4.0);

        let s5_s1 = prod.powf(1.0 / 5.0);
        let s5_s2 = prod.powf(2.0 / 5.0);

        newton_edge[edge_ab].s_4.s1 += s4_s1;
        newton_edge[edge_ab].s_4.s2 += s4_s2;

        newton_edge[edge_ac].s_4.s1 += s4_s1;
        newton_edge[edge_ac].s_4.s2 += s4_s2;

        newton_edge[edge_bc].s_4.s1 += s4_s1;
        newton_edge[edge_bc].s_4.s2 += s4_s2;

        newton_edge[edge_ab].s_5.s1 += s5_s1;
        newton_edge[edge_ab].s_5.s2 += s5_s2;

        newton_edge[edge_ac].s_5.s1 += s5_s1;
        newton_edge[edge_ac].s_5.s2 += s5_s2;

        newton_edge[edge_bc].s_5.s1 += s5_s1;
        newton_edge[edge_bc].s_5.s2 += s5_s2;

        if a < b {
            newton_edge[edge_ab].s3_lower.s1 += weight_ac.powf(1.0 / 3.0);
            newton_edge[edge_ab].s3_lower.s2 += weight_ac.powf(2.0 / 3.0);
            newton_edge[edge_ab].s3_upper.s1 += weight_bc.powf(1.0 / 3.0);
            newton_edge[edge_ab].s3_upper.s2 += weight_bc.powf(2.0 / 3.0);
        } else {
            newton_edge[edge_ab].s3_upper.s1 += weight_ac.powf(1.0 / 3.0);
            newton_edge[edge_ab].s3_upper.s2 += weight_ac.powf(2.0 / 3.0);
            newton_edge[edge_ab].s3_lower.s1 += weight_bc.powf(1.0 / 3.0);
            newton_edge[edge_ab].s3_lower.s2 += weight_bc.powf(2.0 / 3.0);
        }

        if a < c {
            newton_edge[edge_ac].s3_lower.s1 += weight_ab.powf(1.0 / 3.0);
            newton_edge[edge_ac].s3_lower.s2 += weight_ab.powf(2.0 / 3.0);
            newton_edge[edge_ac].s3_upper.s1 += weight_bc.powf(1.0 / 3.0);
            newton_edge[edge_ac].s3_upper.s2 += weight_bc.powf(2.0 / 3.0);
        } else {
            newton_edge[edge_ac].s3_upper.s1 += weight_ab.powf(1.0 / 3.0);
            newton_edge[edge_ac].s3_upper.s2 += weight_ab.powf(2.0 / 3.0);
            newton_edge[edge_ac].s3_lower.s1 += weight_bc.powf(1.0 / 3.0);
            newton_edge[edge_ac].s3_lower.s2 += weight_bc.powf(2.0 / 3.0);
        }

        if b < c {
            newton_edge[edge_bc].s3_lower.s1 += weight_ab.powf(1.0 / 3.0);
            newton_edge[edge_bc].s3_lower.s2 += weight_ab.powf(2.0 / 3.0);
            newton_edge[edge_bc].s3_upper.s1 += weight_ac.powf(1.0 / 3.0);
            newton_edge[edge_bc].s3_upper.s2 += weight_ac.powf(2.0 / 3.0);
        } else {
            newton_edge[edge_bc].s3_upper.s1 += weight_ab.powf(1.0 / 3.0);
            newton_edge[edge_bc].s3_upper.s2 += weight_ab.powf(2.0 / 3.0);
            newton_edge[edge_bc].s3_lower.s1 += weight_ac.powf(1.0 / 3.0);
            newton_edge[edge_bc].s3_lower.s2 += weight_ac.powf(2.0 / 3.0);
        }

        tri_vertex[a] += 1;
        tri_vertex[b] += 1;
        tri_vertex[c] += 1;

        let upper_bound = order_pos.pos[a].min(order_pos.pos[b]).min(order_pos.pos[c]);
        // 4-clique counting
        // forward hashed sorts the adj_list neighbors based on degeneracy ordering so we need to
        // use pos[i] instead of i as key
        common_neighbors_sorted_list_3_by_key(
            &adj_list[a],
            &adj_list[b],
            &adj_list[c],
            &(upper_bound as usize),
            |e| &order_pos.pos[e.node as usize],
            |i, j, k| {
                let common = adj_list[a][i].node;

                let weight_ad = adj_list[a][i].weight;
                let weight_bd = adj_list[b][j].weight;
                let weight_cd = adj_list[c][k].weight;

                let prod = weight_ab * weight_ac * weight_bc * weight_ad * weight_bd * weight_cd;
                k4.count += 1;
                k4.mean_intensity += prod.powf(1.0 / 6.0) as f64;

                diamond_in_k4.non_induced.mean_intensity += ((prod / weight_ab).powf(1.0 / 5.0)
                    + (prod / weight_ac).powf(1.0 / 5.0)
                    + (prod / weight_bc).powf(1.0 / 5.0)
                    + (prod / weight_ad).powf(1.0 / 5.0)
                    + (prod / weight_bd).powf(1.0 / 5.0)
                    + (prod / weight_cd).powf(1.0 / 5.0))
                    as f64;

                c4_in_k4.non_induced.mean_intensity += ((prod / weight_ab / weight_cd)
                    .powf(1.0 / 4.0)
                    + (prod / weight_ac / weight_bd).powf(1.0 / 4.0)
                    + (prod / weight_ad / weight_bc).powf(1.0 / 4.0))
                    as f64;

                let t1 = weight_ab * weight_ac * weight_bc;
                let t2 = weight_ac * weight_ad * weight_cd;
                let t3 = weight_bc * weight_cd * weight_bd;
                let t4 = weight_ab * weight_ad * weight_bd;

                paw_in_k4.non_induced.mean_intensity += ((t1 * weight_bd).powf(1.0 / 4.0)
                    + (t1 * weight_ad).powf(1.0 / 4.0)
                    + (t1 * weight_cd).powf(1.0 / 4.0)
                    + (t2 * weight_ab).powf(1.0 / 4.0)
                    + (t2 * weight_bc).powf(1.0 / 4.0)
                    + (t2 * weight_bd).powf(1.0 / 4.0)
                    + (t3 * weight_ab).powf(1.0 / 4.0)
                    + (t3 * weight_ad).powf(1.0 / 4.0)
                    + (t3 * weight_ac).powf(1.0 / 4.0)
                    + (t4 * weight_ac).powf(1.0 / 4.0)
                    + (t4 * weight_bc).powf(1.0 / 4.0)
                    + (t4 * weight_cd).powf(1.0 / 4.0))
                    as f64;

                star4_in_k4.non_induced.mean_intensity += ((weight_ab * weight_ac * weight_ad)
                    .powf(1.0 / 3.0)
                    + (weight_ab * weight_bc * weight_bd).powf(1.0 / 3.0)
                    + (weight_ac * weight_bc * weight_cd).powf(1.0 / 3.0)
                    + (weight_bd * weight_cd * weight_ad).powf(1.0 / 3.0))
                    as f64;

                let vertical = (weight_ac * weight_bd).powf(1.0 / 3.0);
                let horizontal = (weight_ab * weight_cd).powf(1.0 / 3.0);
                let inner = (weight_ad * weight_bc).powf(1.0 / 3.0);

                path4_in_k4.non_induced.mean_intensity +=
                    ((weight_ad.powf(1.0 / 3.0) + weight_bc.powf(1.0 / 3.0))
                        * (horizontal + vertical)
                        + weight_ab.powf(1.0 / 3.0) * inner
                        + weight_cd.powf(1.0 / 3.0) * inner
                        + weight_ac.powf(1.0 / 3.0) * inner
                        + weight_bd.powf(1.0 / 3.0) * inner
                        + weight_ab.powf(1.0 / 3.0) * vertical
                        + weight_cd.powf(1.0 / 3.0) * vertical
                        + weight_ac.powf(1.0 / 3.0) * horizontal
                        + weight_bd.powf(1.0 / 3.0) * horizontal) as f64;
            },
        );
    });

    // Compute other non-induced counts. Here
    for x in 0..adj_list.n() {
        let deg_x = adj_list[x].len();
        star4.count += deg_x * (deg_x - 1) * (deg_x - 2) / 6;
        star4.mean_intensity += newton_vertex[x].s_3.get_sum_of_products() as f64;

        tailed_triangle.count += tri_vertex[x] * (deg_x - 2);

        let mut y = 0;
        loop {
            if y >= adj_list[x].len() {
                break;
            }
            let neighbor_y = adj_list[x][y].node as usize;
            if order_pos.pos[neighbor_y] >= order_pos.pos[x] {
                break;
            }

            let edge_xy = adj_list[x][y].edge as usize;
            let weight_xy = adj_list[x][y].weight;
            let deg_y = adj_list[neighbor_y].len();

            path4.count += (deg_x - 1) * (deg_y - 1);
            path4.mean_intensity += (weight_xy
                * (neighbors_weight_sum_s1[x] - weight_xy)
                * (neighbors_weight_sum_s1[neighbor_y] - weight_xy))
                as f64;

            tri_edge_count[edge_xy] = max(tri_edge_count[edge_xy], 1);
            diamond.count += tri_edge_count[edge_xy] * (tri_edge_count[edge_xy] - 1) / 2;
            diamond.mean_intensity +=
                (newton_edge[edge_xy].s_5.get_sum_of_products() / weight_xy.powf(1.0 / 5.0)) as f64;

            c4_in_diamond.non_induced.mean_intensity +=
                (newton_edge[edge_xy].s_4.get_sum_of_products() / weight_xy.powf(2.0 / 4.0)) as f64;

            paw_in_diamond.non_induced.mean_intensity +=
                (tri_distal_edge[edge_xy].0.0 * tri_edge_intensity[edge_xy]
                    + tri_distal_edge[edge_xy].1.0 * tri_edge_intensity[edge_xy]
                    - tri_distal_edge[edge_xy].0.1
                    - tri_distal_edge[edge_xy].1.1) as f64;

            star4_in_diamond.non_induced.mean_intensity += (weight_xy.powf(1.0 / 3.0)
                * (newton_edge[edge_xy].s3_upper.get_sum_of_products()
                    + newton_edge[edge_xy].s3_lower.get_sum_of_products()))
                as f64;

            path4_in_diamond.non_induced.mean_intensity += (weight_xy.powf(1.0 / 3.0)
                * (newton_edge[edge_xy].s3_upper.s1 * newton_edge[edge_xy].s3_lower.s1))
                as f64;

            y += 1;
        }
    }
    path4.count -= 3 * triangle.count;
    path4.mean_intensity -= 3.0 * triangle.mean_intensity;

    // c4 are enumerated efficiently. the adj list's neighbors are sorted by degree!!
    (c4.count, c4.mean_intensity) = intensity_c4(&mut adj_list);
    c4.mean_intensity *= c4.count.max(1) as f64; // restore to sum instead of mean

    // converting subgraphlets to induced counts
    //diamond
    diamond_in_k4.induced = diamond_in_k4.non_induced;

    //c4
    c4_in_k4.induced.mean_intensity = c4_in_k4.non_induced.mean_intensity;
    c4_in_diamond.induced.mean_intensity =
        c4_in_diamond.non_induced.mean_intensity - 2.0 * c4_in_k4.induced.mean_intensity;

    // paw
    paw_in_k4.induced.mean_intensity = paw_in_k4.non_induced.mean_intensity;
    paw_in_diamond.induced.mean_intensity =
        paw_in_diamond.non_induced.mean_intensity - 2.0 * paw_in_k4.induced.mean_intensity;

    //star4
    star4_in_k4.induced.mean_intensity = star4_in_k4.non_induced.mean_intensity;
    star4_in_diamond.induced.mean_intensity =
        star4_in_diamond.non_induced.mean_intensity - 3.0 * star4_in_k4.induced.mean_intensity;
    star4_in_paw.induced.mean_intensity = star4_in_paw.non_induced.mean_intensity
        - 2.0 * star4_in_diamond.induced.mean_intensity
        - 3.0 * star4_in_k4.induced.mean_intensity;
    println!(
        "star4 non induced: {}",
        star4_in_k4.non_induced.mean_intensity
    );
    println!("non induced");
    println!("star4_in_k4: {}", star4_in_k4.non_induced.mean_intensity);
    println!(
        "star4_in_diamond: {}",
        star4_in_diamond.non_induced.mean_intensity
    );
    println!("star4_in_paw: {}", star4_in_paw.non_induced.mean_intensity);

    println!("induced");
    println!("star4_in_k4: {}", star4_in_k4.induced.mean_intensity);
    println!(
        "star4_in_diamond: {}",
        star4_in_diamond.induced.mean_intensity
    );
    println!("star4_in_paw: {}", star4_in_paw.induced.mean_intensity);

    // converting to induced counts
    diamond.count -= 6 * k4.count;
    c4.count -= 3 * k4.count + diamond.count;
    tailed_triangle.count -= 12 * k4.count + 4 * diamond.count;
    star4.count -= 4 * k4.count + 2 * diamond.count + tailed_triangle.count;
    path4.count -= 12 * k4.count + 6 * diamond.count + 2 * tailed_triangle.count + 4 * c4.count;

    // converting to induced intensities
    diamond.mean_intensity -= diamond_in_k4.induced.mean_intensity;
    c4.mean_intensity -= c4_in_diamond.induced.mean_intensity + c4_in_k4.induced.mean_intensity;
    tailed_triangle.mean_intensity -=
        paw_in_diamond.induced.mean_intensity + paw_in_k4.induced.mean_intensity;
    star4.mean_intensity -= star4_in_paw.induced.mean_intensity
        + star4_in_diamond.induced.mean_intensity
        + star4_in_k4.induced.mean_intensity;

    k4.mean_intensity /= k4.count.max(1) as f64;
    diamond.mean_intensity /= diamond.count.max(1) as f64;
    c4.mean_intensity /= c4.count.max(1) as f64;
    tailed_triangle.mean_intensity /= tailed_triangle.count.max(1) as f64;
    star4.mean_intensity /= star4.count.max(1) as f64;

    // Add results to the motif stats hashmap
    rv.insert(PATH_4.fingerprint(), path4);
    rv.insert(STAR_4.fingerprint(), star4);

    rv.insert(FOUR_CYCLE.fingerprint(), c4);
    rv.insert(FOUR_CLIQUE.fingerprint(), k4);

    rv.insert(DIAMOND.fingerprint(), diamond);
    rv.insert(TAILED_TRIANGLE.fingerprint(), tailed_triangle);

    return rv;

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
