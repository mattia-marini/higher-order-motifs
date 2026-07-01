use std::ops::AddAssign;

use crate::{
    graph::{AdjList, NodeId},
    misc::{degree_ordering, sort_by_degree},
};

// Orient the adj list such that:
//
// Sort the adjacency list N +(v) of each vertex v, such that the neighborhood
// begins with N −(v) (in arbitrary order), and then is followed by N +(v) (sorted in order of ≺).
fn count_c4_base<W>(adj: &AdjList<W>, order: &[NodeId]) -> usize {
    let mut n_less_count = vec![0; adj.n()];

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

    let mut c_4_count = 0;
    let mut l = vec![0; adj.n()];

    for i in 0..adj.n() {
        let x = order[i] as usize;
        for j in 0..n_less_count[x] {
            let y = adj.adj[x][j].0 as usize;
            let mut k = 0;
            loop {
                let z = adj.adj[y][k].0 as usize;
                if z == x {
                    break;
                }
                c_4_count += l[z];

                l[z] += 1;
                k += 1;
            }
        }

        for j in 0..n_less_count[x] {
            let y = adj.adj[x][j].0 as usize;
            let mut k = 0;
            loop {
                let z = adj.adj[y][k].0 as usize;
                if z == x {
                    break;
                }
                l[z] = 0;
                k += 1;
            }
        }
    }
    c_4_count
}

/// Paul Burkhardt and David G. Harris 4-cycle heuristic
///
/// A mut ref is required for adj list neighbors sorting
pub fn count_c4<W>(adj: &mut AdjList<W>) -> usize {
    let (order, _pos, _max_deg) = sort_by_degree(adj, false);
    count_c4_base(adj, &order)
}

/// Paul Burkhardt and David G. Harris 4-cycle heuristic
///
/// The implementation assumes that the adjacency list is already sorted by degree, so no
/// preprocessing is required. If the adjacency list is not sorted, use `count_c_4()` instead,
/// otherwise the result will be incorrect.
pub fn count_c4_no_sort<W>(adj: &AdjList<W>, order: &[NodeId]) -> usize {
    count_c4_base(adj, order)
}

/// Paul Burkhardt and David G. Harris 4-cycle heuristic
///
/// The implementation assumes that the adjacency list is already sorted by degree, so no
/// preprocessing is required. If the adjacency list is not sorted, use `count_c_4()` instead,
/// otherwise the result will be incorrect.
fn intensity_c4_base<W>(adj: &AdjList<W>, order: &[NodeId]) -> f64
where
    W: num_traits::Float + num_traits::AsPrimitive<f64>,
{
    let mut n_less_count = vec![0; adj.n()];

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

    let mut c_4_count = 0;
    let mut c_4_intensity = 0.0;

    let mut l_count = vec![0; adj.n()];
    let mut l_intensity = vec![0.0; adj.n()];

    for i in 0..adj.n() {
        let x = order[i] as usize;
        for j in 0..n_less_count[x] {
            let y = adj.adj[x][j].0 as usize;
            let w_xy = adj.adj[x][j].1;

            let mut k = 0;
            loop {
                let z = adj.adj[y][k].0 as usize;
                let w_yz = adj.adj[y][k].1;
                if z == x {
                    break;
                }
                c_4_count += l_count[z];
                l_count[z] += 1;

                let curr_part_intensity = (w_xy * w_yz).as_().powf(1.0 / 4.0);
                c_4_intensity += curr_part_intensity * l_intensity[z];
                l_intensity[z] += curr_part_intensity;

                k += 1;
            }
        }

        for j in 0..n_less_count[x] {
            let y = adj.adj[x][j].0 as usize;
            let mut k = 0;
            loop {
                let z = adj.adj[y][k].0 as usize;
                if z == x {
                    break;
                }

                l_count[z] = 0;
                l_intensity[z] = 0.0;
                k += 1;
            }
        }
    }
    c_4_intensity / c_4_count as f64
}

/// Paul Burkhardt and David G. Harris 4-cycle heuristic
///
/// The implementation assumes that the adjacency list is already sorted by degree, so no
/// preprocessing is required.
pub fn intensity_c4<W>(adj: &mut AdjList<W>) -> f64
where
    W: num_traits::Float + num_traits::AsPrimitive<f64>,
{
    let (order, _pos, _max_deg) = sort_by_degree(adj, false);
    intensity_c4_base(adj, &order)
}

/// Paul Burkhardt and David G. Harris 4-cycle heuristic
///
/// The implementation assumes that the adjacency list is already sorted by degree, so no
/// preprocessing is required.
pub fn intensity_c4_no_sort<W>(adj: &AdjList<W>, order: &[NodeId]) -> f64
where
    W: num_traits::Float + num_traits::AsPrimitive<f64>,
{
    intensity_c4_base(adj, order)
}
