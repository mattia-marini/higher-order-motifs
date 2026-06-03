use ouroboros::self_referencing;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Write,
    path::Path,
};

use duplicate::duplicate_item;
use rust_core_macros::{ct_map_accessor, inherent};

use pyo3::{Bound, FromPyObject, PyRef, PyResult, pyclass, pymethods};
use pyo3_stub_gen::{
    PyStubType,
    derive::{gen_stub_pyclass, gen_stub_pymethods},
    impl_stub_type, type_alias,
};

use rkyv::{
    Archive, Deserialize, Serialize, collections::swiss_table::ArchivedHashSet, rend::u32_le,
};

use crate::graph::traits::{HypergraphAccessor, HypergraphBase};

// use super::traits::{HypergraphBase, LiveUnweightedHypergraph, StaticUnweightedHypergraph};
use super::types::*;

use rust_core_macros::ct_map;

#[ct_map(ty(Hx<N, NodeId>), rg(2..6), allocator(Vec<T>))]
#[derive(Archive, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct CtHxVec {}

#[ct_map_accessor(target(CtHxVec))]
pub trait CtHxVecAccessor<const N: usize> {
    #[accessor(&self.buckets.I)]
    #[inline(always)]
    fn get(&self) -> &Vec<Hx<N, NodeId>>;

    #[accessor(&mut self.buckets.I)]
    #[inline(always)]
    fn get_mut(&mut self) -> &mut Vec<Hx<N, NodeId>>;

    #[accessor(self.buckets.I.push(e))]
    #[inline(always)]
    fn push(&mut self, e: Hx<N, NodeId>);

    #[accessor(self.buckets.I.contains(&e))]
    #[inline(always)]
    fn contains(&self, e: &Hx<N, NodeId>) -> bool;
}

#[ct_map(ty(Hx<N, NodeId>), rg(2..6), allocator(HashSet<T>))]
#[derive(Archive, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct CtHxSet {}

#[ct_map_accessor(target(CtHxSet))]
pub trait CtHxSetAccessor<const N: usize> {
    #[accessor(&self.buckets.I)]
    #[inline(always)]
    fn get(&self) -> &HashSet<Hx<N, NodeId>>;

    #[accessor(&mut self.buckets.I)]
    #[inline(always)]
    fn get_mut(&mut self) -> &mut HashSet<Hx<N, NodeId>>;

    #[accessor(self.buckets.I.insert(e))]
    #[inline(always)]
    fn insert(&mut self, e: Hx<N, NodeId>) -> bool;

    #[accessor(self.buckets.I.contains(&e))]
    #[inline(always)]
    fn contains(&self, e: &Hx<N, NodeId>) -> bool;
}

#[derive(Archive, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[gen_stub_pyclass(module = "rust_core.core.graph")]
#[pyclass]
pub struct UnweightedHypergraph {
    pub edge_vec: CtHxVec,
    pub edge_set: CtHxSet,

    pub nodes: HashMap<NodeId, usize>, // track number of edges insisting on a certain node

    #[pyo3(get)]
    n: usize,
    #[pyo3(get)]
    m: usize,
}

#[gen_stub_pymethods(module = "rust_core.core.graph")]
#[pymethods]
impl UnweightedHypergraph {
    #[new]
    pub fn new() -> Self {
        UnweightedHypergraph {
            nodes: HashMap::new(),
            n: 0,
            m: 0,
            edge_vec: CtHxVec::new(),
            edge_set: CtHxSet::new(),
        }
    }

    // #[staticmethod]
    // pub fn from_edges(edges: Vec<Hx>) -> Self {
    //     let mut hg = UnweightedHypergraph::new();
    //     hg.extend_with_edges(edges);
    //     hg
    // }
}

#[duplicate_item(
        graph_type to_native(data);
        [UnweightedHypergraph] [data];
        // [ArchivedUnweightedHypergraph] [data.to_native()];
    )]
#[inherent(
    attr(pymethods),
    attr(gen_stub_pymethods(module = "rust_core.core.graph"))
)]
impl HypergraphBase for graph_type {
    pub fn count_2(&self) -> usize {
        self.edge_vec.get_bucket_2().len()
    }

    pub fn count_3(&self) -> usize {
        self.edge_vec.get_bucket_3().len()
    }

    pub fn count_4(&self) -> usize {
        self.edge_vec.get_bucket_4().len()
    }

    pub fn count_5(&self) -> usize {
        self.edge_vec.get_bucket_5().len()
    }

    pub fn m(&self) -> usize {
        self.edge_vec.tot_size()
    }

    pub fn n(&self) -> usize {
        self.nodes.len()
    }
}

#[duplicate_item(hx_size; [2]; [3]; [4]; [5];)]
impl HypergraphAccessor<hx_size> for UnweightedHypergraph {
    fn count(&self) -> usize {
        CtHxVecAccessor::<2>::get(&self.edge_vec).len()
    }
}

pub fn test() {
    let mut hg = UnweightedHypergraph::new();
    let c = HypergraphAccessor::<5>::count(&hg);
    // hg.count(); // type annotation needed
}

// #[inherent(attr(pymethods), gen_stub_pymethods(module = "rust_core.core.graph"))]
// impl StaticUnweightedHypergraph for UnweightedHypergraph {
//     fn edges(&self) -> Vec<Hx> {
//         todo!()
//     }
//
//     pub fn has_edge(&self, edge: &Hx) -> bool {
//         match edge.len() {
//             2 => self.has_h2(&edge.clone().try_into().unwrap()),
//             3 => self.has_h3(&edge.clone().try_into().unwrap()),
//             4 => self.has_h4(&edge.clone().try_into().unwrap()),
//             5 => self.has_h5(&edge.clone().try_into().unwrap()),
//             _ => self.bigger_edges_set.contains(&edge),
//         }
//     }
//
//     pub fn has_big_edge(&self, edge: &Hx) -> bool {
//         match edge.len() {
//             0..=5 => return false,
//             _ => self.bigger_edges_set.contains(&edge),
//         }
//     }
//
//     pub fn has_h2(&self, edge: &H2) -> bool {
//         self.h2_set.contains(&edge)
//     }
//
//     pub fn has_h3(&self, edge: &H3) -> bool {
//         self.h3_set.contains(&edge)
//     }
//
//     pub fn has_h4(&self, edge: &H4) -> bool {
//         self.h4_set.contains(&edge)
//     }
//
//     pub fn has_h5(&self, edge: &H5) -> bool {
//         self.h5_set.contains(&edge)
//     }
//
//     fn get_h2(&self) -> &[H2] {
//         &self.h2
//     }
//
//     fn get_h3(&self) -> &[H3] {
//         &self.h3
//     }
//
//     fn get_h4(&self) -> &[H4] {
//         &self.h4
//     }
//
//     fn get_h5(&self) -> &[H5] {
//         &self.h5
//     }
// }
//
// #[inherent]
// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// impl LiveUnweightedHypergraph for UnweightedHypergraph {
//     #[inline(always)]
//     pub fn add_edge(&mut self, edge: Hx) {
//         let size = edge.len();
//         match size {
//             2 => self.add_h2(edge.try_into().unwrap()),
//             3 => self.add_h3(edge.try_into().unwrap()),
//             4 => self.add_h4(edge.try_into().unwrap()),
//             5 => self.add_h5(edge.try_into().unwrap()),
//             _ => self.bigger_edges.entry(size).or_insert(vec![]).push(edge),
//         }
//     }
//
//     pub fn extend_with_edges(&mut self, edges: Vec<Hx>) {
//         for edge in edges.into_iter() {
//             self.add_edge(edge);
//         }
//     }
//
//     #[inline(always)]
//     pub fn add_h2(&mut self, edge: H2) {
//         if !self.h2_set.contains(&edge) {
//             self.update_n(&edge);
//             self.h2_set.insert(edge.clone());
//             self.h2.push(edge);
//         }
//     }
//
//     #[inline(always)]
//     pub fn add_h3(&mut self, edge: H3) {
//         if !self.h3_set.contains(&edge) {
//             self.update_n(&edge);
//             self.h3_set.insert(edge.clone());
//             self.h3.push(edge);
//             self.m += 1;
//         }
//     }
//
//     #[inline(always)]
//     pub fn add_h4(&mut self, edge: H4) {
//         if !self.h4_set.contains(&edge) {
//             self.update_n(&edge);
//             self.h4_set.insert(edge.clone());
//             self.h4.push(edge);
//             self.m += 1;
//         }
//     }
//
//     #[inline(always)]
//     pub fn add_h5(&mut self, edge: H5) {
//         if !self.h5_set.contains(&edge) {
//             self.update_n(&edge);
//             self.h5_set.insert(edge.clone());
//             self.h5.push(edge);
//             self.m += 1;
//         }
//     }
//
//     pub fn extend_h2(&mut self, edges: Vec<H2>) {
//         for edge in edges {
//             self.add_h2(edge)
//         }
//     }
//
//     pub fn extend_h3(&mut self, edges: Vec<H3>) {
//         for edge in edges {
//             self.add_h3(edge)
//         }
//     }
//
//     pub fn extend_h4(&mut self, edges: Vec<H4>) {
//         for edge in edges {
//             self.add_h4(edge)
//         }
//     }
//
//     pub fn extend_h5(&mut self, edges: Vec<H5>) {
//         for edge in edges {
//             self.add_h5(edge)
//         }
//     }
//
//     // TODO Should be useless since multiedges are not added in the first place
//     pub fn remove_multiedges(&mut self) {
//         let mut unique_edges = HashSet::new();
//         self.h2.retain(|edge| unique_edges.insert(edge.clone()));
//         let mut unique_edges = HashSet::new();
//         self.h3.retain(|edge| unique_edges.insert(edge.clone()));
//         let mut unique_edges = HashSet::new();
//         self.h4.retain(|edge| unique_edges.insert(edge.clone()));
//         let mut unique_edges = HashSet::new();
//         self.h5.retain(|edge| unique_edges.insert(edge.clone()));
//
//         let mut unique_edges = HashSet::new();
//         for edges in self.bigger_edges.values_mut() {
//             edges.retain(|edge| unique_edges.insert(edge.clone()));
//         }
//     }
// }

impl UnweightedHypergraph {
    #[inline(always)]
    fn update_n<T>(&mut self, edge: &T)
    where
        for<'a> &'a T: IntoIterator<Item = &'a NodeId>,
    {
        for node in edge {
            self.nodes
                .entry(*node)
                .and_modify(|count| *count += 1)
                .or_insert_with(|| {
                    self.n += 1;
                    1
                });
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self)?;

        let mut file = File::create(path)?;
        file.write_all(&bytes)?;

        Ok(())
    }

    pub fn load_from_file_archived<'a, P: AsRef<Path>>(
        path: P,
    ) -> Result<ArchivedUnweightedHypergraphHandle, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;

        let start = std::time::Instant::now();
        let archived = ArchivedUnweightedHypergraphHandle::try_new(bytes, |b| {
            rkyv::access::<ArchivedUnweightedHypergraph, rkyv::rancor::Error>(&b[..])
        })?;
        Ok(archived)
    }

    pub fn load_from_file_deserialized<P: AsRef<Path>>(
        path: P,
    ) -> Result<UnweightedHypergraph, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;

        let start = std::time::Instant::now();
        let archived =
            rkyv::access::<ArchivedUnweightedHypergraph, rkyv::rancor::Error>(&bytes[..])?;
        // let archived = unsafe { rkyv::access_unchecked::<ArchivedFlatAdjList>(&bytes[..]) }; // Faster
        // archived.h2.len(); // Force loading the archived data
        println!("Loading ArchivedHypergraph {:?}", start.elapsed());

        let start = std::time::Instant::now();
        let mut rv = rkyv::deserialize::<UnweightedHypergraph, rkyv::rancor::Error>(archived)?;
        println!("Loading Hypergraph{:?}", start.elapsed());

        Ok(rv)
    }
}

#[self_referencing]
pub struct ArchivedUnweightedHypergraphHandle {
    bytes: Vec<u8>,
    #[borrows(bytes)]
    pub archived: &'this super::ArchivedUnweightedHypergraph,
}

// impl ArchivedUnweightedHypergraphHandle{
//     pub fn new
// }

// All operations that only require read access to the hypergraph should be defined in this trait,
// so that they can be implemented for both `UnweightedHypergraph` and
// `ArchivedUnweightedHypergraph`.
