use ouroboros::self_referencing;
use seq_macro::seq;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Write,
    path::Path,
};

use duplicate::duplicate_item;
use rust_core_macros::{ct_map_accessor, inherent};

use pyo3::{
    Bound, FromPyObject, IntoPyObject, PyRef, PyResult, Python, pyclass, pymethods,
    types::{PyAnyMethods, PySet, PyTuple},
};
use pyo3_stub_gen::{
    PyStubType,
    derive::{gen_stub_pyclass, gen_stub_pymethods},
    impl_stub_type, type_alias,
};

use rkyv::{
    Archive, Deserialize, Serialize, collections::swiss_table::ArchivedHashSet, rend::u32_le,
};

use crate::graph::traits::{
    HxAccessor, PyUnweightedHypergraph, StdHxAccessor, UnweightedHypergraph,
};

use super::types::*;

use rust_core_macros::ct_map;

// Currently unused, set should be enough and easier to maintain.
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
    // pub edge_vec: CtHxVec,
    pub edge_set: CtHxSet,

    pub nodes: HashMap<NodeId, usize>, // track number of edges insisting on a certain node

    #[pyo3(get)]
    n: usize,
    #[pyo3(get)]
    m: usize,
}

// #[gen_stub_pymethods(module = "rust_core.core.graph")]
// #[pymethods]
// impl UnweightedHypergraph {
//     // #[new]
//     // pub fn new() -> Self {
//     //     UnweightedHypergraph {
//     //         nodes: HashMap::new(),
//     //         n: 0,
//     //         m: 0,
//     //         edge_set: CtHxSet::new(),
//     //     }
//     // }
//
//     // #[staticmethod]
//     // pub fn from_edges(edges: Vec<Hx>) -> Self {
//     //     let mut hg = UnweightedHypergraph::new();
//     //     hg.extend_with_edges(edges);
//     //     hg
//     // }
// }

// #[duplicate_item(
//         graph_type to_native(data);
//         [UnweightedHypergraph] [data];
//         // [ArchivedUnweightedHypergraph] [data.to_native()];
//     )]
// #[inherent(
//     attr(pymethods),
//     attr(gen_stub_pymethods(module = "rust_core.core.graph"))
// )]
//

#[duplicate_item(hx_size; [2]; [3]; [4]; [5];)]
impl HxAccessor<hx_size> for UnweightedHypergraph {
    fn count(&self) -> usize {
        CtHxSetAccessor::<hx_size>::get(&self.edge_set).len()
    }

    fn get_bucket(&self) -> &HashSet<Hx<hx_size, NodeId>> {
        self.edge_set.get()
    }

    fn get_bucket_mut(&mut self) -> &mut HashSet<Hx<hx_size, NodeId>> {
        self.edge_set.get_mut()
    }

    fn take_bucket(&mut self) -> HashSet<Hx<hx_size, NodeId>> {
        std::mem::take(self.edge_set.get_mut())
    }

    fn has_hx(&self, edge: Hx<hx_size, NodeId>) -> bool {
        self.edge_set.contains(&edge)
    }

    fn get_hx(&self, edge: Hx<hx_size, NodeId>) -> Option<&Hx<hx_size, NodeId>> {
        self.edge_set.get().get(&edge)
    }

    fn iterate_hx(&self) -> impl Iterator<Item = &Hx<hx_size, NodeId>> {
        self.edge_set.get().iter()
    }

    fn insert_hx(&mut self, edge: Hx<hx_size, NodeId>) -> bool {
        if !self.edge_set.contains(&edge) {
            self.update_n(&edge, true);
            self.edge_set.insert(edge)
        } else {
            false
        }
    }

    fn remove_hx(&mut self, edge: &Hx<hx_size, NodeId>) -> bool {
        if self.edge_set.contains(&edge) {
            self.update_n(&edge, false);
            self.edge_set.get_mut().remove(edge);
            true
        } else {
            false
        }
    }
}

impl StdHxAccessor for UnweightedHypergraph {}

impl UnweightedHypergraph for UnweightedHypergraph {
    fn n(&self) -> usize {
        self.n
    }

    fn m(&self) -> usize {
        self.m
    }

    fn remove_isolated_nodes(&mut self) -> usize {
        let mut removed = 0;
        self.nodes.retain(|_, count| {
            removed += (*count == 0) as usize;
            *count > 0
        });
        removed
    }
}

#[inherent(
    attr(pymethods),
    attr(gen_stub_pymethods(module = "rust_core.core.graph"))
)]
impl PyUnweightedHypergraph for UnweightedHypergraph {
    #[inner(attr(staticmethod))]
    fn new() -> Self {
        Self {
            edge_set: CtHxSet::new(),
            nodes: HashMap::new(),
            n: 0,
            m: 0,
        }
    }

    #[inner(attr(staticmethod))]
    fn from_edges() -> usize {
        todo!()
    }

    fn n(&self) -> usize {
        self.n
    }

    fn m(&self) -> usize {
        self.m
    }

    fn count(&self, order: usize) -> usize {
        match order {
            2 => CtHxSetAccessor::<2>::get(&self.edge_set).len(),
            3 => CtHxSetAccessor::<3>::get(&self.edge_set).len(),
            4 => CtHxSetAccessor::<4>::get(&self.edge_set).len(),
            5 => CtHxSetAccessor::<5>::get(&self.edge_set).len(),
            _ => panic!("Unsupported edge order: {}", order),
        }
    }

    fn edge_vec<'a>(&self, py: Python<'a>, order: usize) -> PyResult<Vec<Bound<'a, PyTuple>>> {
        seq!(N in 2..=5 {
        let rv = match order {
                #(
                    N => CtHxSetAccessor::<N>::get(&self.edge_set)
                        .iter()
                        .map(|e| PyTuple::new(py, e.nodes()))
                        .collect::<PyResult<Vec<_>>>()?,
                )*

            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported edge order: {}", order))),
            };
        });

        Ok(rv)
    }

    fn edge_set<'a>(&self, py: Python<'a>, order: usize) -> PyResult<Bound<'a, PySet>> {
        seq!(N in 2..=5 {
        let rv = match order {
                #(
                    N => CtHxSetAccessor::<N>::get(&self.edge_set)
                        .iter()
                        .map(|e| PyTuple::new(py, e.nodes()))
                        .collect::<PyResult<Vec<_>>>()?,
                )*

            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported edge order: {}", order))),
            };
        });

        Ok(PySet::new(py, rv)?)
    }

    fn has_hx(&self, edge: Bound<'_, PyTuple>) -> PyResult<bool> {
        let order = edge.len()?;

        seq!(N in 2..=5 {
        match order {
                #(
                    N => Ok(CtHxSetAccessor::<N>::get(&self.edge_set).contains(&edge.extract()?)),
                )*

            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported edge order: {}", order))),
            }
        })
    }

    fn get_hx<'a>(
        &self,
        py: Python<'a>,
        edge: Bound<'a, PyTuple>,
    ) -> PyResult<Option<Bound<'a, PyTuple>>> {
        let order = edge.len()?;

        seq!(N in 2..=5 {
        match order {
                #(
                    N => Ok(CtHxSetAccessor::<N>::get(&self.edge_set).get(&edge.extract()?).map(|hx| hx.clone().into_pyobject(py).unwrap())),
                )*

            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported edge order: {}", order))),
            }
        })
    }

    fn insert_hx(&mut self, edge: Bound<'_, PyTuple>) -> PyResult<bool> {
        let order = edge.len()?;

        seq!(N in 2..=5 {
        match order {
                #(
                    N => Ok(HxAccessor::<N>::insert_hx(self, edge.extract()?)),
                )*

            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported edge order: {}", order))),
            }
        })
    }

    fn remove_hx(&mut self, edge: Bound<'_, PyTuple>) -> PyResult<bool> {
        let order = edge.len()?;

        seq!(N in 2..=5 {
        match order {
                #(
                    N => Ok(HxAccessor::<N>::remove_hx(self, &edge.extract()?)),
                )*

            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported edge order: {}", order))),
            }
        })
    }

    fn remove_isolated_nodes(&mut self) -> usize {
        UnweightedHypergraph::remove_isolated_nodes(self)
    }
}

impl UnweightedHypergraph {
    #[inline(always)]
    fn update_n<const N: usize>(&mut self, edge: &Hx<N, NodeId>, add: bool) {
        let increment: i32 = if add { 1 } else { -1 };
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
