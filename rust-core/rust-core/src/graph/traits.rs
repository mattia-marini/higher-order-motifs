use crate::graph::NodeId;

use super::types::{Hx, WHx};
use duplicate::duplicate_item;

use super::unweighted_hypergraph::{ArchivedUnweightedHypergraph, UnweightedHypergraph};

pub trait HypergraphBase: StdSizeHypegraphAccessor {
    fn n(&self) -> usize;
    fn m(&self) -> usize;

    fn count_2(&self) -> usize;
    fn count_3(&self) -> usize;
    fn count_4(&self) -> usize;
    fn count_5(&self) -> usize;

}


pub trait HypergraphAccessor<const N:usize> {
    fn count(&self) -> usize;
}

pub trait StdSizeHypegraphAccessor: HypergraphAccessor<2> + HypergraphAccessor<3> + HypergraphAccessor<4> + HypergraphAccessor<5> {}


#[duplicate_item(
    trait_name hx_type; 
    [StaticUnweightedHypergraph] [Hx]; 
    [StaticWeightedHypergraph] [WHx]
)]
pub trait trait_name: HypergraphBase {
    fn get_h2(&self) -> &[hx_type<2, NodeId>];
    fn get_h3(&self) -> &[hx_type<3, NodeId>];
    fn get_h4(&self) -> &[hx_type<4, NodeId>];
    fn get_h5(&self) -> &[hx_type<5, NodeId>];

    fn has_edge<const N: usize>(&self, edge: &hx_type<N, NodeId>) -> bool;
}

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
