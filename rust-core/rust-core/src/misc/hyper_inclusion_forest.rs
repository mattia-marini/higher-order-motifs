use crate::types::{NodeId, hyperadj_list::HyperAdjList};

/// Returns the inclusion forest of a hypergraph represented as a hyper adjacency list.
pub fn inclusion_forest<W>(
    adj: &HyperAdjList<W>,
    order: Option<(&Vec<NodeId>, &Vec<usize>)>,
) -> Vec<Vec<usize>> {
    // Sorting neighbors by theirs size

    for neighbors in adj {}
    todo!()
}
