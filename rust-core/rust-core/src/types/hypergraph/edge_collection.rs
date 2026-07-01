use foldhash::fast::FixedState;
use rkyv::{Archive, Deserialize, Serialize};
use std::hash::Hash;

use rust_core_macros::{ct_map, ct_map_accessor, hoist_mod, repeat};
use seq_macro::seq;

pub const MIN_HX_SIZE: usize = 2;
pub const MAX_HX_SIZE: usize = 10;
pub const STATIC_BUCKET_SIZE: usize = MAX_HX_SIZE - MIN_HX_SIZE + 1;

pub type HashSet<T> = hashbrown::HashSet<T, FixedState>;

use crate::types::{NodeId, NodeWeight};

use super::hyperedge::Hx;
use crate::types::error::HypergraphError;

#[hoist_mod(attr(repeat(rg(2..11), abs = "N", rel = "I", bucket_name = "__bucket_$N")))]
mod __ {

    #[derive(Archive, Serialize, Deserialize, Clone)]
    pub struct StaticEdgeSet<T, W> {
        #[repeat_item]
        #[doc(hidden)]
        pub(crate) bucket_name: HashSet<Hx<N, T, W>>,
    }

    impl<T, W> StaticEdgeSet<T, W> {
        #[repeat_item(getter_name = "get_bucket_$N")]
        pub fn getter_name(&self) -> &HashSet<Hx<N, T, W>> {
            &self.bucket_name
        }

        #[repeat_item(getter_name = "get_bucket_$N_mut")]
        pub fn getter_name(&mut self) -> &mut HashSet<Hx<N, T, W>> {
            &mut self.bucket_name
        }
    }

    impl<T, W> StaticEdgeSet<T, W> {
        pub fn new() -> Self {
            Self {
                #[repeat_item]
                bucket_name: HashSet::with_hasher(FixedState::default()),
            }
        }
    }
    #[repeat_item]
    impl<T, W> StaticEdgeSetAccessor<N, T, W> for StaticEdgeSet<T, W> {
        fn get_bucket(&self) -> &HashSet<Hx<N, T, W>> {
            &self.bucket_name
        }

        fn get_bucket_mut(&mut self) -> &mut HashSet<Hx<N, T, W>> {
            &mut self.bucket_name
        }

        fn take_bucket(&mut self) -> HashSet<Hx<N, T, W>> {
            std::mem::take(&mut self.bucket_name)
        }
    }
}

pub trait StaticEdgeSetAccessor<const N: usize, T, W> {
    fn get_bucket(&self) -> &HashSet<Hx<N, T, W>>;
    fn get_bucket_mut(&mut self) -> &mut HashSet<Hx<N, T, W>>;
    fn take_bucket(&mut self) -> HashSet<Hx<N, T, W>>;
}

impl<T, W> StaticEdgeSet<T, W> {
    pub fn get_bucket<const N: usize>(&self) -> &HashSet<Hx<N, T, W>>
    where
        Self: StaticEdgeSetAccessor<N, T, W>,
    {
        <Self as StaticEdgeSetAccessor<N, T, W>>::get_bucket(self)
    }

    pub fn get_bucket_mut<const N: usize>(&mut self) -> &mut HashSet<Hx<N, T, W>>
    where
        Self: StaticEdgeSetAccessor<N, T, W>,
    {
        <Self as StaticEdgeSetAccessor<N, T, W>>::get_bucket_mut(self)
    }

    pub fn take_bucket<const N: usize>(&mut self) -> HashSet<Hx<N, T, W>>
    where
        Self: StaticEdgeSetAccessor<N, T, W>,
    {
        <Self as StaticEdgeSetAccessor<N, T, W>>::take_bucket(self)
    }

    pub fn insert<const N: usize>(&mut self, e: Hx<N, T, W>) -> bool
    where
        T: Hash + Eq,
        Self: StaticEdgeSetAccessor<N, T, W>,
    {
        self.get_bucket_mut().insert(e)
    }

    pub fn contains<const N: usize>(&self, nodes: &[T; N]) -> Result<bool, HypergraphError<T>>
    where
        T: PartialEq + Hash + Eq,
        W: Default,
        Self: StaticEdgeSetAccessor<N, T, W>,
    {
        Ok(self.get_bucket().contains(nodes))
    }

    pub fn get_count<const N: usize>(&self) -> usize
    where
        T: Hash + Eq,
        Self: StaticEdgeSetAccessor<N, T, W>,
    {
        self.get_bucket().len()
    }
}

impl<T, W> Default for StaticEdgeSet<T, W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, W> std::fmt::Debug for StaticEdgeSet<T, W>
where
    T: std::fmt::Debug,
    W: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticEdgeSet")
            .field("bucket_2", &self.__bucket_2)
            .field("bucket_3", &self.__bucket_3)
            .field("bucket_4", &self.__bucket_4)
            .field("bucket_5", &self.__bucket_5)
            .field("bucket_6", &self.__bucket_6)
            .field("bucket_7", &self.__bucket_7)
            .field("bucket_8", &self.__bucket_8)
            .field("bucket_9", &self.__bucket_9)
            .field("bucket_10", &self.__bucket_10)
            .finish()
    }
}
