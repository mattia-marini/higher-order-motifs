use crate::graph::AdjList;

#[test]
pub fn test_clique() {
    let mut adj = AdjList::from_edges(vec![
        (0, 1),
        (2, 3),
        (0, 2),
        (1, 3),
        (0, 3),
        (1, 2),
        (0, 4),
        (0, 5),
        (4, 5),
    ]);
    adj.make_undirected();
    adj.sort_neighbors();

    // println!("{:?}", adj.adj);

    // enum_clique_4(&adj, false)
    // println!("{:?}", adj.enum_cliques());
}
