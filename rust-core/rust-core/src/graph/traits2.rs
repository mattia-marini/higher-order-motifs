use std::collections::{HashMap, HashSet};

use crate::graph::NodeId;

use super::types::{Hx, WHx};
use duplicate::duplicate_item;
use pyo3::{Bound, PyResult, Python, types::{PySet, PyTuple}};

use super::unweighted_hypergraph::{ArchivedUnweightedHypergraph, UnweightedHypergraph};


// Rust interface
// Access methods that are related to hyperedge size



// pub trait trait_name<const N:usize> {
// }

#[duplicate_item(
    trait_name      hx_type; 
    [HxAccessor]    [Hx]; 
    [WhxAccessor]   [WHx]
)]
pub trait trait_name<const N:usize> {
    fn count(&self) -> usize;

    fn get_bucket(&self) -> &HashSet<hx_type<N, NodeId>>;
    fn get_bucket_mut(&mut self) -> &mut HashSet<hx_type<N, NodeId>>;
    fn take_bucket(&mut self) -> HashSet<hx_type<N, NodeId>>;

    fn has_hx(&self, edge: Hx<N, NodeId>) -> bool;
    fn get_hx(&self, edge: Hx<N, NodeId>) -> Option<& hx_type<N, NodeId>>;

    fn insert_hx(&mut self, edge: Hx<N, NodeId>) -> bool;

    fn remove_hx(&mut self, edge: &Hx<N, NodeId>) -> bool;

    fn iterate_hx(&self) -> impl Iterator<Item = &hx_type<N, NodeId>>;
    
}


// Access methods that are not related to hyperedge size
#[duplicate_item(
    trait_name                     base_trait       hx_type; 
    [UnweightedHypergraph] [StdHxAccessor]  [Hx]; 
    [WeightedHypergraph]   [StdWHxAccessor] [WHx]
)]
pub trait trait_name: base_trait{
    fn n(&self) -> usize;
    fn m(&self) -> usize;

    fn remove_isolated_nodes(&mut self) -> usize;
}



// Python interface
#[duplicate_item(
    trait_name               hx_type; 
    [PyUnweightedHypergraph] [Hx]; 
    [PyWeightedHypergraph]   [WHx]
)]
pub trait trait_name {
    fn new() -> Self;
    fn from_edges() -> Self;

    fn n(&self) -> usize;
    fn m(&self) -> usize;

    fn count(&self, order: usize) -> usize;

    fn edge_vec<'a>(&self, py: Python<'a>, order: usize) -> PyResult<Vec<Bound<'a, PyTuple>>>;
    fn edge_set<'a>(&self, py: Python<'a>, order: usize) -> PyResult<Bound<'a, PySet>>;

    fn has_hx(&self, edge: Bound<'_, PyTuple>) -> PyResult<bool>;
    fn get_hx<'a>(&self, py: Python<'a>, edge:Bound<'a,PyTuple>) -> PyResult<Option<Bound<'a,PyTuple>>>;

    fn insert_hx(&mut self, edge: Bound<'_, PyTuple>) -> PyResult<bool>;

    fn remove_hx(&mut self, edge: Bound<'_, PyTuple>) -> PyResult<bool>;
    
    fn remove_isolated_nodes(&mut self) -> usize;
}

#[duplicate_item(
    trait_name         base_trait;
    [StdHxAccessor]    [HxAccessor]; 
    [StdWHxAccessor]   [WhxAccessor]
)]
pub trait trait_name: base_trait<2> + base_trait<3> + base_trait<4> + base_trait<5> {}


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
