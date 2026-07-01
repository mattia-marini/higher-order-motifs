use foldhash::fast::FixedState;
use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use rust_core_macros::hoist_mod;
use std::cmp::max;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut, Range, RangeBounds};

use duplicate::duplicate_item;
use rkyv::{Archive, Deserialize, Serialize};

use crate::graph::base::adjacency::ListNeighbor;
use crate::graph::base::{AdjBase, Directed, Directionality, Neighbor, Undirected};
use crate::graph::serialize::DumpCacheToFile;
use crate::graph::{EdgeId, NodeWeight};

use crate::graph::hyperedge::NodeId;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
pub struct AdjList<W, D = Undirected>
where
    D: Directionality,
{
    adj: Vec<Vec<ListNeighbor<NodeId, W>>>,
    n: usize,
    m: usize,

    _directionality_marker: PhantomData<D>,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
pub struct AdjSet<W, D = Undirected>
where
    D: Directionality,
{
    adj: Vec<HashMap<NodeId, W, FixedState>>,
    n: usize,
    m: usize,

    _directionality_marker: PhantomData<D>,
}
