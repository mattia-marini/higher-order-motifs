use crate::graph::AdjList;
use crate::triangle::forward::forward_hashed_cloj;
use crate::graph::types::*;

pub fn count_motifs_3(edges: &(Vec<(NodeId, NodeId)>, Vec<(NodeId, NodeId, NodeId)>)) {
    let adj_list = AdjList::from_edges(&edges.0);
    let mut count_2 = [0, 0]; // star, triangle

    //2 counting
    for neighbors in adj_list.adj.iter() {
        count_2[0] += neighbors.len() * (neighbors.len() - 1) / 2;
    }

    //forward_hashed_cloj(&adj_list.adj, false, |u, v, w| count_2[1] += 1);

    //3 counting
}

pub fn count_motifs_4(
    _edges: &(
        Vec<(usize, usize)>,
        Vec<(usize, usize, usize)>,
        Vec<(usize, usize, usize, usize)>,
    ),
) {
}
