use crate::motifs::base::*;

#[test]
fn test_enum_connected_subgraphs_n2() {
    let graphs = enum_connected_subgraphs(2);
    assert_eq!(graphs.len(), 1);
    assert_eq!(graphs[0], vec![vec![1, 2]]);
}

#[test]
fn test_enum_connected_subgraphs_n3_count() {
    // For n=3: 4 possible hyperedges (one 3-edge + three 2-edges) => 16 subgraphs.
    // Connected spanning ones are 12.
    let graphs = enum_connected_subgraphs(3);
    assert_eq!(graphs.len(), 12);
    assert!(graphs.iter().all(|g| is_connected(g, 3)));
}

#[test]
fn test_canonical_representative_is_in_isomorphisms() {
    let hg: UnweightedHypergraph = vec![vec![1, 2], vec![2, 3]];
    let rep = get_canonical_representative(&hg, Some(3));
    let isos = enum_isomorphisms(&hg, Some(3));
    assert!(isos.contains(&rep));
}

#[test]
fn test_generate_motifs() {
    let motifs3 = generate_motifs(3);
    let motifs4 = generate_motifs(4);
    assert_eq!(motifs3.0.len(), 6);
    assert_eq!(motifs4.0.len(), 171);
}
