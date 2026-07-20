use std::{
    cmp::{max, min},
    error::Error,
    ops::BitOr,
};

use bit_set::BitSet;
use hashbrown::{HashMap, HashSet};
use rust_core::{
    loader::DatasetLoader,
    misc::{
        common_neighbors_sorted_list_3_by_key, cycle::intensity_c4_subinc, degeneracy_ordering,
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
    triangle::forward::forward_sorted_cloj,
    types::{
        NodeId, NodeWeight,
        adj_list::{AdjList, Undirected, WithIncidence},
        hyperadj_list::HyperAdjList,
    },
};
use rust_core_tests::shared::graphlets::STD_HG;
use seq_macro::seq;

#[derive(Debug, Clone, Copy)]
struct EdgeInfo {
    /// the number of triangle incident to the node a
    t_count: usize,

    /// sum of (w(a,b)w(a,d)w(b,d))^(1/3) for each triangle incident to the edge (a, b)
    triangle_13: f32,
    /// sum of (w(a,b)w(a,c)w(b,c))^(1/4) for each triangle incident to the edge (a,b)
    triangle_14: f32,
    /// sum of (w(a,b)w(a,c)w(b,c))^(2/4) for each triangle incident to the edge (a,b)
    triangle_24: f32,

    /// sum of (w(a,b)w(a,c)w(b,c))^(1/5) for each triangle incident to the edge (a,b)
    triangle_15: f32,
    /// sum of (w(a,b)w(a,c)w(b,c))^(2/5) for each triangle incident to the edge (a,b)
    triangle_25: f32,

    /// sum of w(a,d)^(1/3) if a>b, else w(b,d)^(1/3) for each triangle incident to the edge (a, b)
    edge_upper_13: f32,
    /// sum of w(a,d)^(1/3) if a>b, else w(b,d)^(2/3) for each triangle incident to the edge (a, b)
    edge_upper_23: f32,

    /// sum of w(b,d)^(1/3) if a>b, else w(b,d)^(1/3) for each triangle incident to the edge (a, b)
    edge_lower_13: f32,
    /// sum of w(b,d)^(1/3) if a>b, else w(b,d)^(2/3) for each triangle incident to the edge (a, b)
    edge_lower_23: f32,

    /// sum of w(a,d)^(1/3) + w(b,d)^(1/3) for each triangle incident to the edge (a, b)
    distal_edge_sum_13: f32,
    /// sum of w(a,d)^(1/3) + w(b,d)^(1/4) for each triangle incident to the edge (a, b)
    distal_edge_sum_14: f32,

    /// sum of (w(a,b)w(a,d)w(a,d)w(b,d))^(1/3) + (w(a,b)w(a,d)w(b,d)w(b,d))^(1/3) for each triangle incident to the edge (a, b)
    unbalanced_triangle_13: f32,
    /// sum of (w(a,b)w(a,d)w(a,d)w(b,d))^(1/4) + (w(a,b)w(a,d)w(b,d)w(b,d))^(1/3) for each triangle incident to the edge (a, b)
    unbalanced_triangle_14: f32,
}

impl EdgeInfo {
    pub fn empty() -> Self {
        Self {
            t_count: 0,
            triangle_14: 0.,
            triangle_24: 0.,
            triangle_15: 0.,
            triangle_25: 0.,
            edge_upper_13: 0.,
            edge_upper_23: 0.,
            edge_lower_13: 0.,
            edge_lower_23: 0.,
            triangle_13: 0.,
            distal_edge_sum_13: 0.,
            unbalanced_triangle_13: 0.,
            distal_edge_sum_14: 0.,
            unbalanced_triangle_14: 0.,
        }
    }
}

pub fn newton_girard_2(s1: f32, s2: f32) -> f32 {
    (s1 * s1 - s2) / 2.0
}

pub fn newton_girard_3(s1: f32, s2: f32, s3: f32) -> f32 {
    (s1 * s1 * s1 - 3.0 * s1 * s2 + 2.0 * s3) / 6.0
}

#[derive(Debug, Clone, Copy)]
struct NodeInfo {
    /// the number of triangle incident to the node a
    t_count: usize,

    /// sum w(a,b) for each edge (a,b) incident to the node a
    sum_11: f32,
    /// sum w(a,b)^(1/3) for each edge (a,b) incident to the node a
    sum_13: f32,
    /// sum w(a,b)^(2/3) for each edge (a,b) incident to the node a
    sum_23: f32,
    /// sum w(a,b)^(1/4) for each edge (a,b) incident to the node a
    sum_14: f32,
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

/// Helper macro to access the mean intensity of the induced motif stats in a MotifStatsPair
macro_rules! ii {
    ($name:ident) => {
        $name.induced.mean_intensity
    };
}

/// Helper macro to access the mean intensity of the non-induced motif stats in a MotifStatsPair
macro_rules! nii {
    ($name:ident) => {
        $name.non_induced.mean_intensity
    };
}

// pub fn weighted_4(adj: &HyperAdjList<NodeWeight>) -> HashMap<Fingerprint4, MotifStats> {
//     let edges_2 = adj
//         .iter_by_size(2)
//         .map(|(_, e)| (e.nodes[0], e.nodes[1], *e.weight))
//         .collect::<Vec<_>>();
//
//     let (mut adj_list, _direct_map, _inverse_map) =
//         AdjList::<NodeWeight, Undirected, WithIncidence>::from_edges_mapped(edges_2);
//     // adj_list.sort_neighbors();
//     // let adj_set: AdjSet<NodeWeight, Undirected, WithIncidence> = adj_list.clone().into();
//
//     let mut rv = HashMap::new();
//
//     // Final motif stats
//     let mut triangle = MotifStats::new();
//
//     let mut path4 = MotifStats::new();
//     let mut star4 = MotifStats::new();
//
//     let mut k4 = MotifStats::new();
//     let mut c4 = MotifStats::new();
//
//     let mut diamond = MotifStats::new();
//     let mut paw = MotifStats::new();
//
//     // let mut tri_edge_count = vec![0; adj.m()];
//     // let mut tri_edge_intensity = vec![0.0; adj.m()];
//     // let mut tri_vertex = vec![0; adj.n()];
//     // let mut tri_distal_edge = vec![((0.0, 0.0), (0.0, 0.0)); adj.m()];
//
//     // Saving partial stats to convert rom induced to non induced
//     // <motif_a>_in_<motif_b> stores the stats of the non induced occurences of motif_a in motif_b
//
//     // diamon
//     let mut diamond_in_k4 = MotifStatsPair::new();
//
//     // c4
//     let mut c4_in_k4 = MotifStatsPair::new();
//     let mut c4_in_diamond = MotifStatsPair::new();
//
//     // paw
//     let mut paw_in_k4 = MotifStatsPair::new();
//     let mut paw_in_diamond = MotifStatsPair::new();
//
//     // star4
//     let mut star4_in_paw = MotifStatsPair::new();
//     let mut star4_in_diamond = MotifStatsPair::new();
//     let mut star4_in_k4 = MotifStatsPair::new();
//
//     // path4
//     let mut path4_in_paw = MotifStatsPair::new();
//     let mut path4_in_c4 = MotifStatsPair::new();
//
//     let mut path4_in_diamond_ring = MotifStatsPair::new();
//     let mut path4_in_diamond_inner = MotifStatsPair::new();
//
//     let mut path4_in_k4 = MotifStatsPair::new();
//
//     // Coefficient per edge used for fast combinatorial computation of iintensities
//     let mut edge_infos = vec![EdgeInfo::empty(); adj.m()];
//     let mut node_infos = Vec::with_capacity(adj.n());
//
//     for x in 0..adj_list.n() {
//         let mut sum_11 = 0.0;
//         let mut sum_13 = 0.0;
//         let mut sum_23 = 0.0;
//         let mut sum_14 = 0.0;
//
//         for y in adj_list[x].iter() {
//             sum_11 += y.weight;
//             sum_13 += y.weight.powf(1.0 / 3.0);
//             sum_23 += y.weight.powf(2.0 / 3.0);
//             sum_14 += y.weight.powf(1.0 / 4.0);
//         }
//
//         node_infos.push(NodeInfo {
//             t_count: 0,
//             sum_11,
//             sum_13,
//             sum_23,
//             sum_14,
//         });
//     }
//
//     let (mut order_pos, _degeneracy) = degeneracy_ordering(&adj_list);
//     order_pos.reverse();
//
//     // Compute triangles + cliques
//     // Count triangles with forward hashed in O(m^1.5)
//     forward_sorted_cloj(&mut adj_list, Some(&order_pos), |adj_list, t| {
//         let a = t.nodes[0] as usize;
//         let b = t.nodes[1] as usize;
//         let c = t.nodes[2] as usize;
//
//         let edge_ab = t.edges[0] as usize;
//         let edge_ac = t.edges[1] as usize;
//         let edge_bc = t.edges[2] as usize;
//
//         let weight_ab = *t.weights[0];
//         let weight_ac = *t.weights[1];
//         let weight_bc = *t.weights[2];
//
//         let prod = weight_ab * weight_ac * weight_bc;
//
//         triangle.count += 1;
//         triangle.mean_intensity += prod.powf(1.0 / 3.0) as f64;
//
//         edge_infos[edge_ab].t_count += 1;
//         edge_infos[edge_ac].t_count += 1;
//         edge_infos[edge_bc].t_count += 1;
//
//         node_infos[a].t_count += 1;
//         node_infos[b].t_count += 1;
//         node_infos[c].t_count += 1;
//
//         {
//             // for paw counting
//             let prod = prod.powf(1.0 / 4.0);
//             let weight_ab = weight_ab.powf(1.0 / 4.0);
//             let weight_ac = weight_ac.powf(1.0 / 4.0);
//             let weight_bc = weight_bc.powf(1.0 / 4.0);
//
//             edge_infos[edge_ab].distal_edge_sum_14 += weight_ac + weight_bc;
//             edge_infos[edge_ab].unbalanced_triangle_14 += prod * weight_ac + prod * weight_bc;
//
//             edge_infos[edge_ac].distal_edge_sum_14 += weight_ab + weight_bc;
//             edge_infos[edge_ac].unbalanced_triangle_14 += prod * weight_ab + prod * weight_bc;
//
//             edge_infos[edge_bc].distal_edge_sum_14 += weight_ab + weight_ac;
//             edge_infos[edge_bc].unbalanced_triangle_14 += prod * weight_ab + prod * weight_ac;
//
//             paw.mean_intensity += (prod * (node_infos[a].sum_14 - weight_ab - weight_ac)) as f64;
//             paw.mean_intensity += (prod * (node_infos[b].sum_14 - weight_ab - weight_bc)) as f64;
//             paw.mean_intensity += (prod * (node_infos[c].sum_14 - weight_ac - weight_bc)) as f64;
//         }
//
//         {
//             // for star4 counting
//             let prod = prod.powf(1.0 / 3.0);
//             let weight_ab = weight_ab.powf(1.0 / 3.0);
//             let weight_ac = weight_ac.powf(1.0 / 3.0);
//             let weight_bc = weight_bc.powf(1.0 / 3.0);
//
//             edge_infos[edge_ab].distal_edge_sum_13 += weight_ac + weight_bc;
//             edge_infos[edge_ab].unbalanced_triangle_13 += prod * weight_ac + prod * weight_bc;
//
//             edge_infos[edge_ac].distal_edge_sum_13 += weight_ab + weight_bc;
//             edge_infos[edge_ac].unbalanced_triangle_13 += prod * weight_ab + prod * weight_bc;
//
//             edge_infos[edge_bc].distal_edge_sum_13 += weight_ab + weight_ac;
//             edge_infos[edge_bc].unbalanced_triangle_13 += prod * weight_ab + prod * weight_ac;
//
//             nii!(star4_in_paw) +=
//                 ((prod / weight_bc) * (node_infos[a].sum_13 - weight_ab - weight_ac)) as f64;
//             nii!(star4_in_paw) +=
//                 ((prod / weight_ac) * (node_infos[b].sum_13 - weight_ab - weight_bc)) as f64;
//             nii!(star4_in_paw) +=
//                 ((prod / weight_ab) * (node_infos[c].sum_13 - weight_ac - weight_bc)) as f64;
//
//             nii!(path4_in_paw) += (weight_bc
//                 * (weight_ab * (node_infos[a].sum_13 - weight_ab - weight_ac)
//                     + weight_ac * (node_infos[a].sum_13 - weight_ab - weight_ac)))
//                 as f64;
//
//             nii!(path4_in_paw) += (weight_ac
//                 * (weight_ab * (node_infos[b].sum_13 - weight_ab - weight_bc)
//                     + weight_bc * (node_infos[b].sum_13 - weight_ab - weight_bc)))
//                 as f64;
//
//             nii!(path4_in_paw) += (weight_ab
//                 * (weight_ac * (node_infos[c].sum_13 - weight_ac - weight_bc)
//                     + weight_bc * (node_infos[c].sum_13 - weight_ac - weight_bc)))
//                 as f64;
//         }
//
//         let s13 = prod.powf(1.0 / 3.0);
//
//         let s14 = prod.powf(1.0 / 4.0);
//         let s24 = prod.powf(2.0 / 4.0);
//
//         let s15 = prod.powf(1.0 / 5.0);
//         let s25 = prod.powf(2.0 / 5.0);
//
//         edge_infos[edge_ab].triangle_13 += s13;
//         edge_infos[edge_ac].triangle_13 += s13;
//         edge_infos[edge_bc].triangle_13 += s13;
//
//         edge_infos[edge_ab].triangle_14 += s14;
//         edge_infos[edge_ab].triangle_24 += s24;
//
//         edge_infos[edge_ac].triangle_14 += s14;
//         edge_infos[edge_ac].triangle_24 += s24;
//
//         edge_infos[edge_bc].triangle_14 += s14;
//         edge_infos[edge_bc].triangle_24 += s24;
//
//         edge_infos[edge_ab].triangle_15 += s15;
//         edge_infos[edge_ab].triangle_25 += s25;
//
//         edge_infos[edge_ac].triangle_15 += s15;
//         edge_infos[edge_ac].triangle_25 += s25;
//
//         edge_infos[edge_bc].triangle_15 += s15;
//         edge_infos[edge_bc].triangle_25 += s25;
//
//         if a < b {
//             edge_infos[edge_ab].edge_lower_13 += weight_ac.powf(1.0 / 3.0);
//             edge_infos[edge_ab].edge_lower_23 += weight_ac.powf(2.0 / 3.0);
//             edge_infos[edge_ab].edge_upper_13 += weight_bc.powf(1.0 / 3.0);
//             edge_infos[edge_ab].edge_upper_23 += weight_bc.powf(2.0 / 3.0);
//         } else {
//             edge_infos[edge_ab].edge_upper_13 += weight_ac.powf(1.0 / 3.0);
//             edge_infos[edge_ab].edge_upper_23 += weight_ac.powf(2.0 / 3.0);
//             edge_infos[edge_ab].edge_lower_13 += weight_bc.powf(1.0 / 3.0);
//             edge_infos[edge_ab].edge_lower_23 += weight_bc.powf(2.0 / 3.0);
//         }
//
//         if a < c {
//             edge_infos[edge_ac].edge_lower_13 += weight_ab.powf(1.0 / 3.0);
//             edge_infos[edge_ac].edge_lower_23 += weight_ab.powf(2.0 / 3.0);
//             edge_infos[edge_ac].edge_upper_13 += weight_bc.powf(1.0 / 3.0);
//             edge_infos[edge_ac].edge_upper_23 += weight_bc.powf(2.0 / 3.0);
//         } else {
//             edge_infos[edge_ac].edge_upper_13 += weight_ab.powf(1.0 / 3.0);
//             edge_infos[edge_ac].edge_upper_23 += weight_ab.powf(2.0 / 3.0);
//             edge_infos[edge_ac].edge_lower_13 += weight_bc.powf(1.0 / 3.0);
//             edge_infos[edge_ac].edge_lower_23 += weight_bc.powf(2.0 / 3.0);
//         }
//
//         if b < c {
//             edge_infos[edge_bc].edge_lower_13 += weight_ab.powf(1.0 / 3.0);
//             edge_infos[edge_bc].edge_lower_23 += weight_ab.powf(2.0 / 3.0);
//             edge_infos[edge_bc].edge_upper_13 += weight_ac.powf(1.0 / 3.0);
//             edge_infos[edge_bc].edge_upper_23 += weight_ac.powf(2.0 / 3.0);
//         } else {
//             edge_infos[edge_bc].edge_upper_13 += weight_ab.powf(1.0 / 3.0);
//             edge_infos[edge_bc].edge_upper_23 += weight_ab.powf(2.0 / 3.0);
//             edge_infos[edge_bc].edge_lower_13 += weight_ac.powf(1.0 / 3.0);
//             edge_infos[edge_bc].edge_lower_23 += weight_ac.powf(2.0 / 3.0);
//         }
//
//         let upper_bound = order_pos.pos[a].min(order_pos.pos[b]).min(order_pos.pos[c]);
//         // 4-clique counting
//         // forward hashed sorts the adj_list neighbors based on degeneracy ordering so we need to
//         // use pos[i] instead of i as key
//         common_neighbors_sorted_list_3_by_key(
//             &adj_list[a],
//             &adj_list[b],
//             &adj_list[c],
//             &(upper_bound as usize),
//             |e| &order_pos.pos[e.node as usize],
//             |i, j, k| {
//                 // let common = adj_list[a][i].node;
//
//                 let weight_ad = adj_list[a][i].weight;
//                 let weight_bd = adj_list[b][j].weight;
//                 let weight_cd = adj_list[c][k].weight;
//
//                 let prod = weight_ab * weight_ac * weight_bc * weight_ad * weight_bd * weight_cd;
//                 k4.count += 1;
//                 k4.mean_intensity += prod.powf(1.0 / 6.0) as f64;
//
//                 nii!(diamond_in_k4) += ((prod / weight_ab).powf(1.0 / 5.0)
//                     + (prod / weight_ac).powf(1.0 / 5.0)
//                     + (prod / weight_bc).powf(1.0 / 5.0)
//                     + (prod / weight_ad).powf(1.0 / 5.0)
//                     + (prod / weight_bd).powf(1.0 / 5.0)
//                     + (prod / weight_cd).powf(1.0 / 5.0))
//                     as f64;
//
//                 nii!(c4_in_k4) += ((prod / weight_ab / weight_cd).powf(1.0 / 4.0)
//                     + (prod / weight_ac / weight_bd).powf(1.0 / 4.0)
//                     + (prod / weight_ad / weight_bc).powf(1.0 / 4.0))
//                     as f64;
//
//                 let t1 = weight_ab * weight_ac * weight_bc;
//                 let t2 = weight_ac * weight_ad * weight_cd;
//                 let t3 = weight_bc * weight_cd * weight_bd;
//                 let t4 = weight_ab * weight_ad * weight_bd;
//
//                 nii!(paw_in_k4) += ((t1 * weight_bd).powf(1.0 / 4.0)
//                     + (t1 * weight_ad).powf(1.0 / 4.0)
//                     + (t1 * weight_cd).powf(1.0 / 4.0)
//                     + (t2 * weight_ab).powf(1.0 / 4.0)
//                     + (t2 * weight_bc).powf(1.0 / 4.0)
//                     + (t2 * weight_bd).powf(1.0 / 4.0)
//                     + (t3 * weight_ab).powf(1.0 / 4.0)
//                     + (t3 * weight_ad).powf(1.0 / 4.0)
//                     + (t3 * weight_ac).powf(1.0 / 4.0)
//                     + (t4 * weight_ac).powf(1.0 / 4.0)
//                     + (t4 * weight_bc).powf(1.0 / 4.0)
//                     + (t4 * weight_cd).powf(1.0 / 4.0)) as f64;
//
//                 nii!(star4_in_k4) += ((weight_ab * weight_ac * weight_ad).powf(1.0 / 3.0)
//                     + (weight_ab * weight_bc * weight_bd).powf(1.0 / 3.0)
//                     + (weight_ac * weight_bc * weight_cd).powf(1.0 / 3.0)
//                     + (weight_bd * weight_cd * weight_ad).powf(1.0 / 3.0))
//                     as f64;
//
//                 let vertical = (weight_ac * weight_bd).powf(1.0 / 3.0);
//                 let horizontal = (weight_ab * weight_cd).powf(1.0 / 3.0);
//                 let inner = (weight_ad * weight_bc).powf(1.0 / 3.0);
//
//                 nii!(path4_in_k4) += ((weight_ad.powf(1.0 / 3.0) + weight_bc.powf(1.0 / 3.0))
//                     * (horizontal + vertical)
//                     + weight_ab.powf(1.0 / 3.0) * inner
//                     + weight_cd.powf(1.0 / 3.0) * inner
//                     + weight_ac.powf(1.0 / 3.0) * inner
//                     + weight_bd.powf(1.0 / 3.0) * inner
//                     + weight_ab.powf(1.0 / 3.0) * vertical
//                     + weight_cd.powf(1.0 / 3.0) * vertical
//                     + weight_ac.powf(1.0 / 3.0) * horizontal
//                     + weight_bd.powf(1.0 / 3.0) * horizontal)
//                     as f64;
//             },
//         );
//     });
//
//     // Compute other non-induced counts. Here
//     for x in 0..adj_list.n() {
//         let deg_x = adj_list[x].len();
//         star4.count += deg_x * (deg_x - 1) * (deg_x - 2) / 6;
//         star4.mean_intensity += newton_girard_3(
//             node_infos[x].sum_13,
//             node_infos[x].sum_23,
//             node_infos[x].sum_11,
//         ) as f64;
//
//         paw.count += node_infos[x].t_count * (deg_x - 2);
//
//         let mut y = 0;
//         loop {
//             if y >= adj_list[x].len() {
//                 break;
//             }
//             let neighbor_y = adj_list[x][y].node as usize;
//             if order_pos.pos[neighbor_y] >= order_pos.pos[x] {
//                 break;
//             }
//
//             let edge_xy = adj_list[x][y].edge as usize;
//             let weight_xy = adj_list[x][y].weight;
//             let deg_y = adj_list[neighbor_y].len();
//
//             path4.count += (deg_x - 1) * (deg_y - 1);
//             path4.mean_intensity += (weight_xy.powf(1.0 / 3.0)
//                 * (node_infos[x].sum_13 - weight_xy.powf(1.0 / 3.0))
//                 * (node_infos[neighbor_y].sum_13 - weight_xy.powf(1.0 / 3.0)))
//                 as f64;
//
//             edge_infos[edge_xy].t_count = max(edge_infos[edge_xy].t_count, 1);
//             diamond.count += edge_infos[edge_xy].t_count * (edge_infos[edge_xy].t_count - 1) / 2;
//             diamond.mean_intensity += (newton_girard_2(
//                 edge_infos[edge_xy].triangle_15,
//                 edge_infos[edge_xy].triangle_25,
//             ) / weight_xy.powf(1.0 / 5.0)) as f64;
//
//             nii!(c4_in_diamond) += (newton_girard_2(
//                 edge_infos[edge_xy].triangle_14,
//                 edge_infos[edge_xy].triangle_24,
//             ) / weight_xy.powf(2.0 / 4.0)) as f64;
//
//             nii!(paw_in_diamond) +=
//                 (edge_infos[edge_xy].triangle_14 * edge_infos[edge_xy].distal_edge_sum_14
//                     - edge_infos[edge_xy].unbalanced_triangle_14) as f64;
//
//             nii!(star4_in_diamond) += (weight_xy.powf(1.0 / 3.0)
//                 * (newton_girard_2(
//                     edge_infos[edge_xy].edge_upper_13,
//                     edge_infos[edge_xy].edge_upper_23,
//                 ) + newton_girard_2(
//                     edge_infos[edge_xy].edge_lower_13,
//                     edge_infos[edge_xy].edge_lower_23,
//                 ))) as f64;
//
//             nii!(path4_in_diamond_ring) += ((edge_infos[edge_xy].triangle_13
//                 * edge_infos[edge_xy].distal_edge_sum_13
//                 - edge_infos[edge_xy].unbalanced_triangle_13)
//                 / weight_xy.powf(1.0 / 3.0)) as f64;
//
//             // can collapse into triangles but they are subtracted later
//             nii!(path4_in_diamond_inner) += (weight_xy.powf(1.0 / 3.0)
//                 * (edge_infos[edge_xy].edge_upper_13)
//                 * (edge_infos[edge_xy].edge_lower_13))
//                 as f64;
//             y += 1;
//         }
//     }
//     path4.count -= 3 * triangle.count;
//     path4.mean_intensity -= 3.0 * triangle.mean_intensity;
//     nii!(path4_in_diamond_inner) -= 3.0 * triangle.mean_intensity;
//     println!("triangle count: {}", triangle.count);
//     println!("triangle mean intensity: {}", triangle.mean_intensity);
//
//     // c4 are enumerated efficiently. the adj list's neighbors are sorted by degree!!
//     (c4.count, c4.mean_intensity, nii!(path4_in_c4)) = intensity_c4_subinc(&mut adj_list);
//     c4.mean_intensity *= c4.count.max(1) as f64; // restore to sum instead of mean
//     nii!(path4_in_c4) *= 4.0 * c4.count.max(1) as f64;
//
//     // converting subgraphlets to induced counts
//     // diamond
//     ii!(diamond_in_k4) = nii!(diamond_in_k4);
//
//     // c4
//     ii!(c4_in_k4) = nii!(c4_in_k4);
//     ii!(c4_in_diamond) = nii!(c4_in_diamond) - 2.0 * ii!(c4_in_k4);
//
//     // paw
//     ii!(paw_in_k4) = nii!(paw_in_k4);
//     ii!(paw_in_diamond) = nii!(paw_in_diamond) - 2.0 * ii!(paw_in_k4);
//
//     // star4
//     ii!(star4_in_k4) = nii!(star4_in_k4);
//     ii!(star4_in_diamond) = nii!(star4_in_diamond) - 3.0 * ii!(star4_in_k4);
//     ii!(star4_in_paw) = nii!(star4_in_paw) - 2.0 * ii!(star4_in_diamond) - 3.0 * ii!(star4_in_k4);
//
//     // path4
//     ii!(path4_in_k4) = nii!(path4_in_k4);
//     // ii!(path4_in_diamond) = nii!(path4_in_diamond) - 3.0 * ii!(path4_in_k4);
//     ii!(path4_in_diamond_ring) = nii!(path4_in_diamond_ring) - 2.0 * ii!(path4_in_k4);
//     ii!(path4_in_diamond_inner) = nii!(path4_in_diamond_inner) - ii!(path4_in_k4);
//     ii!(path4_in_c4) = nii!(path4_in_c4) - ii!(path4_in_diamond_ring) - ii!(path4_in_k4);
//     ii!(path4_in_paw) = nii!(path4_in_paw)
//         - ii!(path4_in_diamond_ring)
//         - 2.0 * ii!(path4_in_diamond_inner)
//         - 2.0 * ii!(path4_in_k4);
//
//     // converting to induced counts
//     diamond.count -= 6 * k4.count;
//     c4.count -= 3 * k4.count + diamond.count;
//     paw.count -= 12 * k4.count + 4 * diamond.count;
//     star4.count -= 4 * k4.count + 2 * diamond.count + paw.count;
//     path4.count -= 12 * k4.count + 6 * diamond.count + 2 * paw.count + 4 * c4.count;
//
//     // converting to induced intensities
//     diamond.mean_intensity -= ii!(diamond_in_k4);
//     c4.mean_intensity -= ii!(c4_in_diamond) + ii!(c4_in_k4);
//     paw.mean_intensity -= ii!(paw_in_diamond) + ii!(paw_in_k4);
//     star4.mean_intensity -= ii!(star4_in_paw) + ii!(star4_in_diamond) + ii!(star4_in_k4);
//     path4.mean_intensity -= ii!(path4_in_paw)
//         + ii!(path4_in_c4)
//         + ii!(path4_in_diamond_ring)
//         + ii!(path4_in_diamond_inner)
//         + ii!(path4_in_k4);
//
//     // Add results to the motif stats hashmap
//
//     if path4.count > 0 {
//         rv.insert(PATH_4.fingerprint(), path4);
//     }
//     if star4.count > 0 {
//         rv.insert(STAR_4.fingerprint(), star4);
//     }
//
//     if c4.count > 0 {
//         rv.insert(FOUR_CYCLE.fingerprint(), c4);
//     }
//     if k4.count > 0 {
//         rv.insert(FOUR_CLIQUE.fingerprint(), k4);
//     }
//
//     if diamond.count > 0 {
//         rv.insert(DIAMOND.fingerprint(), diamond);
//     }
//     if paw.count > 0 {
//         rv.insert(TAILED_TRIANGLE.fingerprint(), paw);
//     }
//
//     let mut edges_2 = HashMap::with_capacity(adj.count_by_size(2));
//     let mut edges_3 = HashMap::with_capacity(adj.count_by_size(3));
//     let mut hyper_diamonds: HashMap<(NodeId, NodeId), Vec<NodeId>> =
//         HashMap::with_capacity(adj.m() * 3);
//
//     for (_edge_id, edge) in adj.iter_by_size(2) {
//         edges_2.insert((edge.nodes[0], edge.nodes[1]), edge.weight);
//     }
//
//     for (_edge_id, edge) in adj.iter_by_size(3) {
//         edges_3.insert((edge.nodes[0], edge.nodes[1], edge.nodes[2]), edge.weight);
//     }
//
//     for (_edge_id, edge) in adj.iter_by_size(3) {
//         let nodes = edge.nodes;
//         const AVG_SIZE: usize = 10;
//
//         hyper_diamonds
//             .entry((nodes[0], nodes[1]))
//             .and_modify(|v| v.push(nodes[2]))
//             .or_insert_with(|| {
//                 let mut rv = Vec::with_capacity(AVG_SIZE);
//                 rv.push(nodes[2]);
//                 rv
//             });
//
//         hyper_diamonds
//             .entry((nodes[0], nodes[2]))
//             .and_modify(|v| v.push(nodes[1]))
//             .or_insert_with(|| {
//                 let mut rv = Vec::with_capacity(AVG_SIZE);
//                 rv.push(nodes[1]);
//                 rv
//             });
//
//         hyper_diamonds
//             .entry((nodes[1], nodes[2]))
//             .and_modify(|v| v.push(nodes[0]))
//             .or_insert_with(|| {
//                 let mut rv = Vec::with_capacity(AVG_SIZE);
//                 rv.push(nodes[0]);
//                 rv
//             });
//     }
//
//     println!("{:?}", edges_2);
//     println!("{:?}", edges_3);
//     println!("{:?}", hyper_diamonds);
//
//     macro_rules! check_nodes {
//     ($motif:ident, $intensity:ident, $edges: expr, $nodes: ident, [ $($n:expr),+ ]) => {
//         // This expands to a reference to a tuple: &(nodes[0], nodes[1], ...)
//         if let Some(&weight) = $edges.get(&( $( $nodes[$n] ),+ )) {
//             $motif.add_edge_with_nodes({ CompressedNodeSet::from_array([ $( $n as u8 ),+ ]) });
//             // c2 += 1;
//             $intensity *= weight;
//         }
//         };
//     }
//
//     macro_rules! check_nodes_const {
//     ($motif:ident, $intensity:ident, $edges: expr, $nodes: expr, [ $($n:expr),+ ]) => {
//         // This expands to a reference to a tuple: &(nodes[0], nodes[1], ...)
//         if let Some(&weight) = $edges.get(&( $( $nodes[$n] ),+ )) {
//             $motif.add_edge_with_nodes(const { CompressedNodeSet::from_array([ $( $n as u8),+ ]) });
//             // c2 += 1;
//             $intensity *= weight;
//         }
//      };
//     }
//
//     // #[inline(always)]
//     // /// Inputs MUST satisfy: a < b and c < d
//     // pub fn merge_sorted_pairs(a: u32, b: u32, c: u32, d: u32) -> [u32; 4] {
//     //     let (min1, max1) = if a < c { (a, c) } else { (c, a) };
//     //     let (min2, max2) = if b < d { (b, d) } else { (d, b) };
//     //
//     //     let (mid1, mid2) = if max1 < min2 {
//     //         (max1, min2)
//     //     } else {
//     //         (min2, max1)
//     //     };
//     //
//     //     [min1, mid1, mid2, max2]
//     // }
//
//     // let mut groups4
//
//     for (&(a, b), extension_nodes) in hyper_diamonds.iter() {
//         let mut motif2 = CompactMotif::<4>::zero();
//         let mut motif3 = CompactMotif::<4>::zero();
//         let mut i2 = 1.0;
//         let mut i3 = 1.0;
//
//         for i in 0..extension_nodes.len() {
//             for j in (i + 1)..extension_nodes.len() {
//                 let c = extension_nodes[i];
//                 let d = extension_nodes[j];
//                 // let (c, d) = if c < d { (c, d) } else { (d, c) };
//                 let sorted = {
//                     let mut rv = [c, a, b, d];
//                     for i in 1..3 {
//                         if rv[i] < rv[i - 1] {
//                             rv.swap(i, i - 1);
//                         } else {
//                             break;
//                         }
//                     }
//                     for i in (0..3).rev() {
//                         if rv[i] > rv[i + 1] {
//                             rv.swap(i, i + 1);
//                         } else {
//                             break;
//                         }
//                     }
//                     rv
//                 };
//
//                 check_nodes!(motif3, i3, edges_3, sorted, [0, 1, 2]);
//                 check_nodes!(motif3, i3, edges_3, sorted, [0, 1, 3]);
//                 check_nodes!(motif3, i3, edges_3, sorted, [0, 2, 3]);
//                 check_nodes!(motif3, i3, edges_3, sorted, [1, 2, 3]);
//
//                 if motif3.edge_count() == 4 && a < min(c, d) {
//                     continue;
//                 }
//
//                 println!("{} - {}", a, b);
//
//                 check_nodes_const!(motif2, i2, edges_2, sorted, [0, 1]);
//                 check_nodes_const!(motif2, i2, edges_2, sorted, [0, 2]);
//                 check_nodes_const!(motif2, i2, edges_2, sorted, [0, 3]);
//                 check_nodes_const!(motif2, i2, edges_2, sorted, [1, 2]);
//                 check_nodes_const!(motif2, i2, edges_2, sorted, [1, 3]);
//                 check_nodes_const!(motif2, i2, edges_2, sorted, [2, 3]);
//
//                 let c2 = motif2.edge_count();
//                 let c3 = motif3.edge_count();
//
//                 rv.entry(motif2.fingerprint())
//                     .and_modify(|stats: &mut MotifStats| {
//                         stats.count -= 1;
//                         stats.mean_intensity -= i2.powf(1.0 / c2 as f32) as f64;
//                     });
//
//                 let entry = rv
//                     .entry(motif2.bitor(motif3).fingerprint())
//                     .or_insert(MotifStats::new());
//                 entry.count += 1;
//                 entry.mean_intensity -= (i2 * i3).powf(1.0 / (c2 + c3) as f32) as f64;
//             }
//         }
//     }
//
//     for (_edge_id, edge) in adj.iter_by_size(4) {
//         let motif_edge4 = const {
//             let mut rv = CompactMotif::<4>::zero();
//             rv.const_add_edge_with_nodes(CompressedNodeSet::from_array([0, 1, 2, 3]));
//             rv
//         };
//         let mut motif = CompactMotif::<4>::zero();
//         let mut intensity = 1.0;
//
//         check_nodes_const!(motif, intensity, edges_2, edge.nodes, [0, 1]);
//         check_nodes_const!(motif, intensity, edges_2, edge.nodes, [0, 2]);
//         check_nodes_const!(motif, intensity, edges_2, edge.nodes, [0, 3]);
//         check_nodes_const!(motif, intensity, edges_2, edge.nodes, [1, 2]);
//         check_nodes_const!(motif, intensity, edges_2, edge.nodes, [1, 3]);
//         check_nodes_const!(motif, intensity, edges_2, edge.nodes, [2, 3]);
//
//         check_nodes_const!(motif, intensity, edges_3, edge.nodes, [0, 1, 2]);
//         check_nodes_const!(motif, intensity, edges_3, edge.nodes, [0, 1, 3]);
//         check_nodes_const!(motif, intensity, edges_3, edge.nodes, [0, 2, 3]);
//         check_nodes_const!(motif, intensity, edges_3, edge.nodes, [1, 2, 3]);
//
//         let count = motif.edge_count();
//
//         rv.entry(motif.fingerprint())
//             .and_modify(|stats: &mut MotifStats| {
//                 stats.count -= 1;
//                 stats.mean_intensity += (intensity).powf(1.0 / (count) as f32) as f64;
//             });
//
//         motif.add_edge_with_nodes(const { CompressedNodeSet::from_array([0, 1, 2, 3]) });
//         let entry = rv
//             .entry(motif.bitor(motif_edge4).fingerprint())
//             .or_insert(MotifStats::new());
//         entry.count += 1;
//         entry.mean_intensity += (intensity * edge.weight).powf(1.0 / (count + 1) as f32) as f64;
//     }
//
//     for (_fingerprint, stats) in rv.iter_mut() {
//         stats.mean_intensity /= stats.count.max(1) as f64;
//     }
//
//     rv.retain(|_f, v| v.count > 0);
//
//     rv
// }

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
    let mut paw = MotifStats::new();

    // let mut tri_edge_count = vec![0; adj.m()];
    // let mut tri_edge_intensity = vec![0.0; adj.m()];
    // let mut tri_vertex = vec![0; adj.n()];
    // let mut tri_distal_edge = vec![((0.0, 0.0), (0.0, 0.0)); adj.m()];

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
    let mut path4_in_diamond_ring = MotifStatsPair::new();
    let mut path4_in_diamond_inner = MotifStatsPair::new();

    let mut path4_in_k4 = MotifStatsPair::new();

    // Coefficient per edge used for fast combinatorial computation of iintensities
    let mut edge_infos = vec![EdgeInfo::empty(); adj.m()];
    let mut node_infos = Vec::with_capacity(adj.n());

    for x in 0..adj_list.n() {
        let mut sum_11 = 0.0;
        let mut sum_13 = 0.0;
        let mut sum_23 = 0.0;
        let mut sum_14 = 0.0;

        for y in adj_list[x].iter() {
            sum_11 += y.weight;
            sum_13 += y.weight.powf(1.0 / 3.0);
            sum_23 += y.weight.powf(2.0 / 3.0);
            sum_14 += y.weight.powf(1.0 / 4.0);
        }

        node_infos.push(NodeInfo {
            t_count: 0,
            sum_11,
            sum_13,
            sum_23,
            sum_14,
        });
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

        edge_infos[edge_ab].t_count += 1;
        edge_infos[edge_ac].t_count += 1;
        edge_infos[edge_bc].t_count += 1;

        node_infos[a].t_count += 1;
        node_infos[b].t_count += 1;
        node_infos[c].t_count += 1;

        {
            // for paw counting
            let prod = prod.powf(1.0 / 4.0);
            let weight_ab = weight_ab.powf(1.0 / 4.0);
            let weight_ac = weight_ac.powf(1.0 / 4.0);
            let weight_bc = weight_bc.powf(1.0 / 4.0);

            edge_infos[edge_ab].distal_edge_sum_14 += weight_ac + weight_bc;
            edge_infos[edge_ab].unbalanced_triangle_14 += prod * weight_ac + prod * weight_bc;

            edge_infos[edge_ac].distal_edge_sum_14 += weight_ab + weight_bc;
            edge_infos[edge_ac].unbalanced_triangle_14 += prod * weight_ab + prod * weight_bc;

            edge_infos[edge_bc].distal_edge_sum_14 += weight_ab + weight_ac;
            edge_infos[edge_bc].unbalanced_triangle_14 += prod * weight_ab + prod * weight_ac;

            paw.mean_intensity += (prod * (node_infos[a].sum_14 - weight_ab - weight_ac)) as f64;
            paw.mean_intensity += (prod * (node_infos[b].sum_14 - weight_ab - weight_bc)) as f64;
            paw.mean_intensity += (prod * (node_infos[c].sum_14 - weight_ac - weight_bc)) as f64;
        }

        {
            // for star4 counting
            let prod = prod.powf(1.0 / 3.0);
            let weight_ab = weight_ab.powf(1.0 / 3.0);
            let weight_ac = weight_ac.powf(1.0 / 3.0);
            let weight_bc = weight_bc.powf(1.0 / 3.0);

            edge_infos[edge_ab].distal_edge_sum_13 += weight_ac + weight_bc;
            edge_infos[edge_ab].unbalanced_triangle_13 += prod * weight_ac + prod * weight_bc;

            edge_infos[edge_ac].distal_edge_sum_13 += weight_ab + weight_bc;
            edge_infos[edge_ac].unbalanced_triangle_13 += prod * weight_ab + prod * weight_bc;

            edge_infos[edge_bc].distal_edge_sum_13 += weight_ab + weight_ac;
            edge_infos[edge_bc].unbalanced_triangle_13 += prod * weight_ab + prod * weight_ac;

            nii!(star4_in_paw) +=
                ((prod / weight_bc) * (node_infos[a].sum_13 - weight_ab - weight_ac)) as f64;
            nii!(star4_in_paw) +=
                ((prod / weight_ac) * (node_infos[b].sum_13 - weight_ab - weight_bc)) as f64;
            nii!(star4_in_paw) +=
                ((prod / weight_ab) * (node_infos[c].sum_13 - weight_ac - weight_bc)) as f64;

            nii!(path4_in_paw) += (weight_bc
                * (weight_ab * (node_infos[a].sum_13 - weight_ab - weight_ac)
                    + weight_ac * (node_infos[a].sum_13 - weight_ab - weight_ac)))
                as f64;

            nii!(path4_in_paw) += (weight_ac
                * (weight_ab * (node_infos[b].sum_13 - weight_ab - weight_bc)
                    + weight_bc * (node_infos[b].sum_13 - weight_ab - weight_bc)))
                as f64;

            nii!(path4_in_paw) += (weight_ab
                * (weight_ac * (node_infos[c].sum_13 - weight_ac - weight_bc)
                    + weight_bc * (node_infos[c].sum_13 - weight_ac - weight_bc)))
                as f64;
        }

        let s13 = prod.powf(1.0 / 3.0);

        let s14 = prod.powf(1.0 / 4.0);
        let s24 = prod.powf(2.0 / 4.0);

        let s15 = prod.powf(1.0 / 5.0);
        let s25 = prod.powf(2.0 / 5.0);

        edge_infos[edge_ab].triangle_13 += s13;
        edge_infos[edge_ac].triangle_13 += s13;
        edge_infos[edge_bc].triangle_13 += s13;

        edge_infos[edge_ab].triangle_14 += s14;
        edge_infos[edge_ab].triangle_24 += s24;

        edge_infos[edge_ac].triangle_14 += s14;
        edge_infos[edge_ac].triangle_24 += s24;

        edge_infos[edge_bc].triangle_14 += s14;
        edge_infos[edge_bc].triangle_24 += s24;

        edge_infos[edge_ab].triangle_15 += s15;
        edge_infos[edge_ab].triangle_25 += s25;

        edge_infos[edge_ac].triangle_15 += s15;
        edge_infos[edge_ac].triangle_25 += s25;

        edge_infos[edge_bc].triangle_15 += s15;
        edge_infos[edge_bc].triangle_25 += s25;

        if a < b {
            edge_infos[edge_ab].edge_lower_13 += weight_ac.powf(1.0 / 3.0);
            edge_infos[edge_ab].edge_lower_23 += weight_ac.powf(2.0 / 3.0);
            edge_infos[edge_ab].edge_upper_13 += weight_bc.powf(1.0 / 3.0);
            edge_infos[edge_ab].edge_upper_23 += weight_bc.powf(2.0 / 3.0);
        } else {
            edge_infos[edge_ab].edge_upper_13 += weight_ac.powf(1.0 / 3.0);
            edge_infos[edge_ab].edge_upper_23 += weight_ac.powf(2.0 / 3.0);
            edge_infos[edge_ab].edge_lower_13 += weight_bc.powf(1.0 / 3.0);
            edge_infos[edge_ab].edge_lower_23 += weight_bc.powf(2.0 / 3.0);
        }

        if a < c {
            edge_infos[edge_ac].edge_lower_13 += weight_ab.powf(1.0 / 3.0);
            edge_infos[edge_ac].edge_lower_23 += weight_ab.powf(2.0 / 3.0);
            edge_infos[edge_ac].edge_upper_13 += weight_bc.powf(1.0 / 3.0);
            edge_infos[edge_ac].edge_upper_23 += weight_bc.powf(2.0 / 3.0);
        } else {
            edge_infos[edge_ac].edge_upper_13 += weight_ab.powf(1.0 / 3.0);
            edge_infos[edge_ac].edge_upper_23 += weight_ab.powf(2.0 / 3.0);
            edge_infos[edge_ac].edge_lower_13 += weight_bc.powf(1.0 / 3.0);
            edge_infos[edge_ac].edge_lower_23 += weight_bc.powf(2.0 / 3.0);
        }

        if b < c {
            edge_infos[edge_bc].edge_lower_13 += weight_ab.powf(1.0 / 3.0);
            edge_infos[edge_bc].edge_lower_23 += weight_ab.powf(2.0 / 3.0);
            edge_infos[edge_bc].edge_upper_13 += weight_ac.powf(1.0 / 3.0);
            edge_infos[edge_bc].edge_upper_23 += weight_ac.powf(2.0 / 3.0);
        } else {
            edge_infos[edge_bc].edge_upper_13 += weight_ab.powf(1.0 / 3.0);
            edge_infos[edge_bc].edge_upper_23 += weight_ab.powf(2.0 / 3.0);
            edge_infos[edge_bc].edge_lower_13 += weight_ac.powf(1.0 / 3.0);
            edge_infos[edge_bc].edge_lower_23 += weight_ac.powf(2.0 / 3.0);
        }

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

                nii!(diamond_in_k4) += ((prod / weight_ab).powf(1.0 / 5.0)
                    + (prod / weight_ac).powf(1.0 / 5.0)
                    + (prod / weight_bc).powf(1.0 / 5.0)
                    + (prod / weight_ad).powf(1.0 / 5.0)
                    + (prod / weight_bd).powf(1.0 / 5.0)
                    + (prod / weight_cd).powf(1.0 / 5.0))
                    as f64;

                nii!(c4_in_k4) += ((prod / weight_ab / weight_cd).powf(1.0 / 4.0)
                    + (prod / weight_ac / weight_bd).powf(1.0 / 4.0)
                    + (prod / weight_ad / weight_bc).powf(1.0 / 4.0))
                    as f64;

                let t1 = weight_ab * weight_ac * weight_bc;
                let t2 = weight_ac * weight_ad * weight_cd;
                let t3 = weight_bc * weight_cd * weight_bd;
                let t4 = weight_ab * weight_ad * weight_bd;

                nii!(paw_in_k4) += ((t1 * weight_bd).powf(1.0 / 4.0)
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
                    + (t4 * weight_cd).powf(1.0 / 4.0)) as f64;

                nii!(star4_in_k4) += ((weight_ab * weight_ac * weight_ad).powf(1.0 / 3.0)
                    + (weight_ab * weight_bc * weight_bd).powf(1.0 / 3.0)
                    + (weight_ac * weight_bc * weight_cd).powf(1.0 / 3.0)
                    + (weight_bd * weight_cd * weight_ad).powf(1.0 / 3.0))
                    as f64;

                let vertical = (weight_ac * weight_bd).powf(1.0 / 3.0);
                let horizontal = (weight_ab * weight_cd).powf(1.0 / 3.0);
                let inner = (weight_ad * weight_bc).powf(1.0 / 3.0);

                nii!(path4_in_k4) += ((weight_ad.powf(1.0 / 3.0) + weight_bc.powf(1.0 / 3.0))
                    * (horizontal + vertical)
                    + weight_ab.powf(1.0 / 3.0) * inner
                    + weight_cd.powf(1.0 / 3.0) * inner
                    + weight_ac.powf(1.0 / 3.0) * inner
                    + weight_bd.powf(1.0 / 3.0) * inner
                    + weight_ab.powf(1.0 / 3.0) * vertical
                    + weight_cd.powf(1.0 / 3.0) * vertical
                    + weight_ac.powf(1.0 / 3.0) * horizontal
                    + weight_bd.powf(1.0 / 3.0) * horizontal)
                    as f64;
            },
        );
    });

    // Compute other non-induced counts. Here
    for x in 0..adj_list.n() {
        let deg_x = adj_list[x].len();
        star4.count += deg_x * (deg_x - 1) * (deg_x - 2) / 6;
        star4.mean_intensity += newton_girard_3(
            node_infos[x].sum_13,
            node_infos[x].sum_23,
            node_infos[x].sum_11,
        ) as f64;

        paw.count += node_infos[x].t_count * (deg_x - 2);

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
            path4.mean_intensity += (weight_xy.powf(1.0 / 3.0)
                * (node_infos[x].sum_13 - weight_xy.powf(1.0 / 3.0))
                * (node_infos[neighbor_y].sum_13 - weight_xy.powf(1.0 / 3.0)))
                as f64;

            edge_infos[edge_xy].t_count = max(edge_infos[edge_xy].t_count, 1);
            diamond.count += edge_infos[edge_xy].t_count * (edge_infos[edge_xy].t_count - 1) / 2;
            diamond.mean_intensity += (newton_girard_2(
                edge_infos[edge_xy].triangle_15,
                edge_infos[edge_xy].triangle_25,
            ) / weight_xy.powf(1.0 / 5.0)) as f64;

            nii!(c4_in_diamond) += (newton_girard_2(
                edge_infos[edge_xy].triangle_14,
                edge_infos[edge_xy].triangle_24,
            ) / weight_xy.powf(2.0 / 4.0)) as f64;

            nii!(paw_in_diamond) +=
                (edge_infos[edge_xy].triangle_14 * edge_infos[edge_xy].distal_edge_sum_14
                    - edge_infos[edge_xy].unbalanced_triangle_14) as f64;

            nii!(star4_in_diamond) += (weight_xy.powf(1.0 / 3.0)
                * (newton_girard_2(
                    edge_infos[edge_xy].edge_upper_13,
                    edge_infos[edge_xy].edge_upper_23,
                ) + newton_girard_2(
                    edge_infos[edge_xy].edge_lower_13,
                    edge_infos[edge_xy].edge_lower_23,
                ))) as f64;

            nii!(path4_in_diamond_ring) += ((edge_infos[edge_xy].triangle_13
                * edge_infos[edge_xy].distal_edge_sum_13
                - edge_infos[edge_xy].unbalanced_triangle_13)
                / weight_xy.powf(1.0 / 3.0)) as f64;

            // can collapse into triangles but they are subtracted later
            nii!(path4_in_diamond_inner) += (weight_xy.powf(1.0 / 3.0)
                * (edge_infos[edge_xy].edge_upper_13)
                * (edge_infos[edge_xy].edge_lower_13))
                as f64;
            y += 1;
        }
    }
    path4.count -= 3 * triangle.count;
    path4.mean_intensity -= 3.0 * triangle.mean_intensity;
    nii!(path4_in_diamond_inner) -= 3.0 * triangle.mean_intensity;
    println!("triangle count: {}", triangle.count);
    println!("triangle mean intensity: {}", triangle.mean_intensity);

    // c4 are enumerated efficiently. the adj list's neighbors are sorted by degree!!
    (c4.count, c4.mean_intensity, nii!(path4_in_c4)) = intensity_c4_subinc(&mut adj_list);
    c4.mean_intensity *= c4.count.max(1) as f64; // restore to sum instead of mean
    nii!(path4_in_c4) *= 4.0 * c4.count.max(1) as f64;

    // converting subgraphlets to induced counts
    // diamond
    ii!(diamond_in_k4) = nii!(diamond_in_k4);

    // c4
    ii!(c4_in_k4) = nii!(c4_in_k4);
    ii!(c4_in_diamond) = nii!(c4_in_diamond) - 2.0 * ii!(c4_in_k4);

    // paw
    ii!(paw_in_k4) = nii!(paw_in_k4);
    ii!(paw_in_diamond) = nii!(paw_in_diamond) - 2.0 * ii!(paw_in_k4);

    // star4
    ii!(star4_in_k4) = nii!(star4_in_k4);
    ii!(star4_in_diamond) = nii!(star4_in_diamond) - 3.0 * ii!(star4_in_k4);
    ii!(star4_in_paw) = nii!(star4_in_paw) - 2.0 * ii!(star4_in_diamond) - 3.0 * ii!(star4_in_k4);

    // path4
    nii!(path4_in_diamond) = nii!(path4_in_diamond_ring) + nii!(path4_in_diamond_inner);
    ii!(path4_in_k4) = nii!(path4_in_k4);
    // ii!(path4_in_diamond) = nii!(path4_in_diamond) - 3.0 * ii!(path4_in_k4);
    ii!(path4_in_diamond_ring) = nii!(path4_in_diamond_ring) - 2.0 * ii!(path4_in_k4);
    ii!(path4_in_diamond_inner) = nii!(path4_in_diamond_inner) - ii!(path4_in_k4);
    ii!(path4_in_c4) = nii!(path4_in_c4) - ii!(path4_in_diamond_ring) - ii!(path4_in_k4);
    ii!(path4_in_paw) = nii!(path4_in_paw)
        - ii!(path4_in_diamond_ring)
        - 2.0 * ii!(path4_in_diamond_inner)
        - 2.0 * ii!(path4_in_k4);

    // converting to induced counts
    diamond.count -= 6 * k4.count;
    c4.count -= 3 * k4.count + diamond.count;
    paw.count -= 12 * k4.count + 4 * diamond.count;
    star4.count -= 4 * k4.count + 2 * diamond.count + paw.count;
    path4.count -= 12 * k4.count + 6 * diamond.count + 2 * paw.count + 4 * c4.count;

    // converting to induced intensities
    diamond.mean_intensity -= ii!(diamond_in_k4);
    c4.mean_intensity -= ii!(c4_in_diamond) + ii!(c4_in_k4);
    paw.mean_intensity -= ii!(paw_in_diamond) + ii!(paw_in_k4);
    star4.mean_intensity -= ii!(star4_in_paw) + ii!(star4_in_diamond) + ii!(star4_in_k4);
    path4.mean_intensity -= ii!(path4_in_paw)
        + ii!(path4_in_c4)
        + ii!(path4_in_diamond_ring)
        + ii!(path4_in_diamond_inner)
        + ii!(path4_in_k4);

    // Add results to the motif stats hashmap
    if path4.count > 0 {
        rv.insert(PATH_4.fingerprint(), path4);
    }
    if star4.count > 0 {
        rv.insert(STAR_4.fingerprint(), star4);
    }

    if c4.count > 0 {
        rv.insert(FOUR_CYCLE.fingerprint(), c4);
    }
    if k4.count > 0 {
        rv.insert(FOUR_CLIQUE.fingerprint(), k4);
    }

    if diamond.count > 0 {
        rv.insert(DIAMOND.fingerprint(), diamond);
    }
    if paw.count > 0 {
        rv.insert(TAILED_TRIANGLE.fingerprint(), paw);
    }

    let mut groups4: HashSet<[NodeId; 4]> =
        HashSet::with_capacity(adj.count_by_size(3) + adj.count_by_size(4));

    let mut extension_nodes_map = vec![(CompactMotif::<4>::zero(), 1.0, 1.0); adj.m()];
    let mut extension_nodes_list = vec![0; adj.n()];
    let mut inserted = BitSet::with_capacity(adj.n());

    for (pivot_edge_id, pivot_edge) in adj.iter_by_size(3) {
        extension_nodes_list.clear();
        let mut center_motif = CompactMotif::<4>::zero();
        let mut center_intensity_2 = 1.0;
        center_motif.add_edge_with_nodes(CompressedNodeSet::from_array([0, 1, 2]));
        for curr_pivot in 0..3 {
            for (_edge_id, edge) in adj.iter_incident_by_size(pivot_edge.nodes[curr_pivot], 2) {
                assert!(edge.nodes.len() == 2);

                let non_pivot = if edge.nodes[0] == pivot_edge.nodes[curr_pivot] {
                    edge.nodes[1]
                } else {
                    edge.nodes[0]
                };

                let (is_inner, inner_index) = if non_pivot == pivot_edge.nodes[0] {
                    (true, 0)
                } else if non_pivot == pivot_edge.nodes[1] {
                    (true, 1)
                } else if non_pivot == pivot_edge.nodes[2] {
                    (true, 2)
                } else {
                    (false, 0)
                };

                if is_inner {
                    if pivot_edge.nodes[curr_pivot] < non_pivot {
                        center_motif.add_edge_with_nodes(CompressedNodeSet::from_array([
                            curr_pivot as u8,
                            inner_index,
                        ]));
                        center_intensity_2 *= edge.weight;
                    }
                } else {
                    if !inserted.contains(non_pivot as usize) {
                        inserted.insert(non_pivot as usize);
                        extension_nodes_list.push(non_pivot);
                    }

                    extension_nodes_map[non_pivot as usize]
                        .0
                        .add_edge_with_nodes(CompressedNodeSet::from_array([curr_pivot as u8, 3]));

                    extension_nodes_map[non_pivot as usize].1 *= edge.weight;
                }
            }

            for (edge_id, edge) in adj.iter_incident_by_size(pivot_edge.nodes[curr_pivot], 3) {
                assert!(edge.nodes.len() == 3);
                if pivot_edge_id == edge_id {
                    continue;
                }

                let mut outer = [0; 2];
                let mut inner = [(0, 0); 2];
                let mut outer_count = 0;
                let mut inner_count = 0;
                for i in 0..3 {
                    if edge.nodes[i] == pivot_edge.nodes[0] {
                        inner[inner_count] = (edge.nodes[i], 0);
                        inner_count += 1;
                    } else if edge.nodes[i] == pivot_edge.nodes[1] {
                        inner[inner_count] = (edge.nodes[i], 1);
                        inner_count += 1;
                    } else if edge.nodes[i] == pivot_edge.nodes[2] {
                        inner[inner_count] = (edge.nodes[i], 2);
                        inner_count += 1;
                    } else {
                        outer[outer_count] = edge.nodes[i];
                        outer_count += 1;
                    }
                }

                if outer_count == 1 {
                    let outer = outer[0];
                    // let pivot = pivot_edge.nodes[i];
                    let (inner_node, inner_index) = if inner[0].0 == pivot_edge.nodes[curr_pivot] {
                        inner[1]
                    } else {
                        inner[0]
                    };

                    if pivot_edge.nodes[curr_pivot] < inner_node {
                        continue;
                    }

                    if !inserted.contains(outer as usize) {
                        inserted.insert(outer as usize);
                        extension_nodes_list.push(outer);
                    }

                    extension_nodes_map[outer as usize].0.add_edge_with_nodes(
                        CompressedNodeSet::from_array([curr_pivot as u8, inner_index, 3]),
                    );

                    extension_nodes_map[outer as usize].2 *= edge.weight;
                }
            }

            // if edge.nodes.len() == 2 {
            // } else if edge.nodes.len() == 3 {
            //     let outer = 0;
            //     for j in 0..3 {
            //         if edge.nodes[j] != pivot_edge.nodes[i] && edge.nodes[j] {
            //             continue;
            //         }
            //     }
            //
            //     let outer = {
            // };
            // }
        }

        for &outer in &extension_nodes_list {
            let sorted_group4 = {
                let mut v = [
                    outer,
                    pivot_edge.nodes[0],
                    pivot_edge.nodes[1],
                    pivot_edge.nodes[2],
                ];
                for i in 1..4 {
                    if v[i] < v[i - 1] {
                        v.swap(i, i - 1);
                    }
                }
                v
            };

            if groups4.contains(&sorted_group4) {
                extension_nodes_map[outer as usize] = (CompactMotif::<4>::zero(), 1.0, 1.0);
                inserted.remove(outer as usize);
                continue;
            }

            let peripheral_motif = extension_nodes_map[outer as usize].0;
            let c2 = center_motif.filtered_by_order(2).edge_count()
                + extension_nodes_map[outer as usize]
                    .0
                    .filtered_by_order(2)
                    .edge_count();

            let c3 = extension_nodes_map[outer as usize]
                .0
                .filtered_by_order(3)
                .edge_count()
                + 1;

            let i2 = extension_nodes_map[outer as usize].1 * center_intensity_2;
            let i3 = extension_nodes_map[outer as usize].2 * pivot_edge.weight;

            // correcting overcounting
            rv.entry(
                center_motif
                    .bitor(peripheral_motif)
                    .filtered_by_order(2)
                    .fingerprint(),
            )
            .and_modify(|e| {
                e.count -= 1;
                e.mean_intensity -= i2.powf(1.0 / c2 as f32) as f64;
            });

            // adding actual count
            let entry = rv
                .entry(center_motif.bitor(peripheral_motif).fingerprint())
                .or_insert(MotifStats::new());
            entry.count += 1;
            entry.mean_intensity += (i2 * i3).powf(1.0 / (c2 + c3) as f32) as f64;

            extension_nodes_map[outer as usize] = (CompactMotif::<4>::zero(), 1.0, 1.0);
            inserted.remove(outer as usize);

            groups4.insert(sorted_group4);
        }
    }

    let mut edges_2 = HashMap::with_capacity(adj.count_by_size(2));

    let mut edges_3 = HashMap::with_capacity(adj.count_by_size(3));

    for (_edge_id, edge) in adj.iter_by_size(2) {
        edges_2.insert(edge.nodes, *edge.weight);
    }

    for (_edge_id, edge) in adj.iter_by_size(3) {
        edges_3.insert(edge.nodes, *edge.weight);
    }

    for (_edge_id, edge) in adj.iter_by_size(4) {
        let mut inner_intensity = 1.0;

        let mut motif = CompactMotif::<4>::zero();
        for i in 0..4 {
            for j in (i + 1)..4 {
                if let Some(&weight) = edges_2.get([edge.nodes[i], edge.nodes[j]].as_slice()) {
                    motif.add_edge_with_nodes(CompressedNodeSet::from_array([i as u8, j as u8]));
                    inner_intensity *= weight as f32;
                }
            }
        }

        for i in 0..4 {
            for j in (i + 1)..4 {
                for k in (j + 1)..4 {
                    if let Some(&weight) =
                        edges_3.get([edge.nodes[i], edge.nodes[j], edge.nodes[k]].as_slice())
                    {
                        motif.add_edge_with_nodes(CompressedNodeSet::from_array([
                            i as u8, j as u8, k as u8,
                        ]));

                        inner_intensity *= weight as f32;
                    }
                }
            }
        }

        // fix overcounting
        rv.entry(motif.fingerprint()).and_modify(|stats| {
            stats.count -= 1;
            stats.mean_intensity -= inner_intensity.powf(1.0 / motif.edge_count() as f32) as f64;
        });

        // add motif
        motif.add_edge_with_nodes(CompressedNodeSet::from_array([0, 1, 2, 3]));
        let stats = rv.entry(motif.fingerprint()).or_insert(MotifStats::new());
        stats.count += 1;
        stats.mean_intensity +=
            (inner_intensity * edge.weight).powf(1.0 / motif.edge_count() as f32) as f64;
    }

    for (_fingerprint, stats) in rv.iter_mut() {
        stats.mean_intensity /= stats.count.max(1) as f64;
    }

    rv.retain(|_f, v| v.count > 0);

    rv
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let mut hg = DatasetLoader::builder()
        .cached(false)
        .hospital()
        .weighted()
        .load()?;
    hg.normalize_node_ids();

    seq!(N in 5..11 { hg.take_edges::<N>(); });
    let (adj, _, _) = HyperAdjList::<NodeWeight>::from_hypergraph_mapped(hg.0);

    // let (_hg, adj) = STD_HG.clone();

    let t = std::time::Instant::now();
    println!("n: {}, m: {}", adj.n(), adj.m());
    let rv = weighted_4(&adj);
    println!("Finished in: {:?}", t.elapsed());

    println!("len {}", rv.len());
    for (_number, (motif, stats)) in rv.iter().enumerate() {
        println!(
            "{}\t{}\t{}",
            stats.count,
            stats.mean_intensity,
            motif.get_canonical_rep()
        );
    }

    Ok(())
}
