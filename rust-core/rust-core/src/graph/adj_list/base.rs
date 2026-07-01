use rkyv::{Archive, Deserialize, Serialize};
use std::{hash::Hash, marker::PhantomData};

use crate::graph::NodeId;

pub trait Directionality {}
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Clone, Copy)]
pub struct Directed;
impl Directionality for Directed {}
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Clone, Copy)]
pub struct Undirected;
impl Directionality for Undirected {}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
pub struct Neighbor<N, E, W> {
    /// The node ID
    pub node: N,
    /// The edge ID associated with this neighbor.
    pub edge: E,
    /// The weight of the edge connecting to this neighbor.
    pub weight: W,
}

impl<N, E, W> Neighbor<N, E, W> {
    pub fn new(node: N, weight: W, edge: E) -> Self {
        Self { node, weight, edge }
    }
}

pub trait AdjBase<W> {
    fn add_edge(&mut self, u: NodeId, v: NodeId, w: W)
    where
        W: Clone;
    fn remove_edges_between(&mut self, u: NodeId, v: NodeId);
}

pub mod incidence {
    use rkyv::{Archive, Deserialize, Serialize};
    use std::{hash::Hash, marker::PhantomData};

    /// A neighbor node ID, used for storing in a vec.
    #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
    pub struct ListNeighbor<N, E, W> {
        /// The node ID
        pub node: N,
        /// The edge ID associated with this neighbor.
        pub edge: E,
        /// The weight of the edge connecting to this neighbor.
        pub weight: W,
    }

    impl<N, E, W> ListNeighbor<N, E, W> {
        pub fn new(node: N, weight: W, edge: E) -> Self {
            Self { node, weight, edge }
        }
    }

    /// A neighbor with no node ID, used for storing in a set.
    #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
    pub struct SetNeighbor<E, W> {
        /// The edge ID associated with this neighbor.
        pub edge: E,
        /// The weight of the edge connecting to this neighbor.
        pub weight: W,
    }

    impl<E, W> SetNeighbor<E, W> {
        pub fn new(weight: W, edge: E) -> Self {
            Self { weight, edge }
        }
    }
}

pub mod adjacency {
    use rkyv::{Archive, Deserialize, Serialize};
    use std::{hash::Hash, marker::PhantomData};

    /// A neighbor node ID, used for storing in a vec.
    #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
    pub struct ListNeighbor<N, W> {
        /// The node ID
        pub node: N,
        /// The weight of the edge connecting to this neighbor.
        pub weight: W,
    }

    impl<N, W> ListNeighbor<N, W> {
        pub fn new(node: N, weight: W) -> Self {
            Self { node, weight }
        }
    }
}
