// use super::types::{H2, H3, H4, H5, WHx};
// use super::unweighted_hypergraph::{ArchivedUnweightedHypergraph, UnweightedHypergraph};
//
// pub trait HypergraphBase {
//     fn n(&self) -> usize;
//     fn m(&self) -> usize;
//
//     fn count_2(&self) -> usize;
//     fn count_3(&self) -> usize;
//     fn count_4(&self) -> usize;
//     fn count_5(&self) -> usize;
// }
//
// pub trait StaticUnweightedHypergraph: HypergraphBase {
//     fn edges(&self) -> Vec<WHx>;
//
//     fn get_h2(&self) -> &[H2];
//     fn get_h3(&self) -> &[H3];
//     fn get_h4(&self) -> &[H4];
//     fn get_h5(&self) -> &[H5];
//
//     fn has_edge(&self, edge: &WHx) -> bool;
//     fn has_big_edge(&self, edge: &WHx) -> bool;
//     fn has_h2(&self, edge: &H2) -> bool;
//     fn has_h3(&self, edge: &H3) -> bool;
//     fn has_h4(&self, edge: &H4) -> bool;
//     fn has_h5(&self, edge: &H5) -> bool;
// }
//
// /// All operations that require mutable access to the hypergraph should be defined in this trait
// pub trait LiveUnweightedHypergraph: StaticUnweightedHypergraph {
//     fn add_edge(&mut self, edge: WHx);
//     fn extend_with_edges(&mut self, edges: Vec<WHx>);
//
//     fn add_h2(&mut self, edge: H2);
//     fn add_h3(&mut self, edge: H3);
//     fn add_h4(&mut self, edge: H4);
//     fn add_h5(&mut self, edge: H5);
//
//     fn extend_h2(&mut self, edges: Vec<H2>);
//     fn extend_h3(&mut self, edges: Vec<H3>);
//     fn extend_h4(&mut self, edges: Vec<H4>);
//     fn extend_h5(&mut self, edges: Vec<H5>);
//
//     fn remove_multiedges(&mut self);
// }
//
// // pub trait StaticWeightedHypergraph {
// // }
