use std::collections::HashMap;

use crate::graph::AdjList;

pub fn count_motifs_3(edges: &(Vec<(usize, usize)>, Vec<(usize, usize, usize)>)) {
    let adj_list = AdjList::from_edges(&edges.0);
    let mut count_2 = [0, 0]; // star, triangle

    //2 counting
    for (_n, neighbors) in adj_list.adj.iter().enumerate() {
        count_2[0] += neighbors.len() * (neighbors.len() - 1) / 2;
    }

    //3 counting
}

pub fn count_motifs_4(
    edges: &(
        Vec<(usize, usize)>,
        Vec<(usize, usize, usize)>,
        Vec<(usize, usize, usize, usize)>,
    ),
) {
}
