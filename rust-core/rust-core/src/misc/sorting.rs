use num_traits::{AsPrimitive, PrimInt};

use crate::types::hyperadj_list::{HyperAdjList, HyperAdjListBase};
use crate::types::{Hx, Hypergraph, NodeId};
use std::borrow::Cow;
use std::hash::Hash;

use crate::types::adj_list::AdjList;
use crate::types::adj_list::traits::{Direction, Incidence};
use crate::types::adj_list::{
    adj_list::AdjListBase,
    traits::{AdjConfig, NeighborContainer},
};

/// Returns a degree ordering of the vertices, the position of each vertex in that ordering, and
/// the maximum degree of the graph.
/// Time Complexity: O(n)
pub fn degree_ordering<C: AdjConfig>(
    adj: &AdjListBase<C>,
    decreasing: bool,
) -> (OrderAndPos, usize) {
    let n = adj.n();
    if n == 0 {
        return (OrderAndPos::empty(), 0);
    }

    let deg: Vec<usize> = adj
        .iter_neighbors()
        .map(|neighbors| neighbors.len())
        .collect();
    let max_deg = *deg.iter().max().unwrap_or(&0);

    let mut bin_count = vec![0; max_deg + 1];
    for &d in &deg {
        bin_count[d] += 1;
    }

    let mut start_pos = 0;
    let mut bin_starts = vec![0; max_deg + 1];
    for d in 0..=max_deg {
        bin_starts[d] = start_pos;
        start_pos += bin_count[d];
    }

    let mut order = vec![0; n];
    let mut pos = vec![0; n];

    if decreasing {
        for v in (0..n).rev() {
            let d = deg[v];
            pos[v] = bin_starts[d];
            order[bin_starts[d]] = v as NodeId;
            bin_starts[d] += 1;
        }
    } else {
        for v in 0..n {
            let d = deg[v];
            pos[v] = bin_starts[d];
            order[bin_starts[d]] = v as NodeId;
            bin_starts[d] += 1;
        }
    }

    (OrderAndPos::new(order, pos, true), max_deg)
}

/// Sorts the neighbors of each vertex in the adjacency list by the following conditions:
/// u ≺ v if deg(u) < deg(v); if deg(u) = deg(v) the tie breaker is arbitrary
///
/// Time Complexity: O(e log d), where e is the number of edges and d is the maximum degree.
pub fn sort_by_degree<W, D: Direction, I: Incidence>(
    adj: &mut AdjList<W, D, I>,
    _decreasing: bool,
) -> (OrderAndPos, usize) {
    let (order_pos, max_deg) = degree_ordering(adj, false);

    for v in 0..adj.n() {
        adj[v].sort_by_key(|neighbor| order_pos.pos[neighbor.node as usize]);
    }

    (order_pos, max_deg)
}

/// Returns a degeneracy ordering of the graph, the position of each vertex,
/// and the degeneracy (k) of the graph.
/// Complexity: O(n + m)
pub fn degeneracy_ordering<C: AdjConfig>(adj: &AdjListBase<C>) -> (OrderAndPos, usize) {
    let n = adj.n();
    if n == 0 {
        return (OrderAndPos::empty(), 0);
    }

    let mut deg: Vec<usize> = adj
        .iter_neighbors()
        .map(|neighbors| neighbors.len())
        .collect();
    let max_deg = *deg.iter().max().unwrap_or(&0);

    let mut bin_count = vec![0; max_deg + 1];
    for &d in &deg {
        bin_count[d] += 1;
    }

    let mut bin_starts = vec![0; max_deg + 1];
    let mut start_pos = 0;
    for d in 0..=max_deg {
        bin_starts[d] = start_pos;
        start_pos += bin_count[d];
    }

    let mut temp_starts = bin_starts.clone();
    let mut order = vec![0; n];
    let mut pos = vec![0; n];
    for v in 0..n {
        pos[v] = temp_starts[deg[v]];
        order[pos[v]] = v as NodeId;
        temp_starts[deg[v]] += 1;
    }

    let mut k = 0;
    macro_rules! decrease_node {
        ($node:expr) => {{
            unsafe {
                let n = $node;
                let u_deg = *deg.get_unchecked(n);
                let u_pos = *pos.get_unchecked(n);

                let first_node_pos = *bin_starts.get_unchecked(u_deg);
                let first_node = *order.get_unchecked(first_node_pos);

                if n as NodeId != first_node {
                    let tmp = *pos.get_unchecked(first_node as usize);
                    *pos.get_unchecked_mut(first_node as usize) = *pos.get_unchecked(n);
                    *pos.get_unchecked_mut(n) = tmp;

                    let tmp = *order.get_unchecked(first_node_pos);
                    *order.get_unchecked_mut(first_node_pos) = *order.get_unchecked(u_pos);
                    *order.get_unchecked_mut(u_pos) = tmp;
                }

                *bin_starts.get_unchecked_mut(u_deg) += 1;
                *deg.get_unchecked_mut(n) -= 1;
            }
        }};
    }

    for i in 0..n {
        let v = order[i] as usize;
        k = std::cmp::max(k, deg[v]);

        for neighbor in adj[v].iter_neighbors() {
            let u = *neighbor.node as usize;
            if pos[u] > i {
                decrease_node!(u);
                decrease_node!(v);
            }
        }
    }

    (OrderAndPos::new(order, pos, true), k)
}

/// Returns a degeneracy ordering of the hypergraph, the position of each vertex,
/// and the degeneracy (k) of the hypergraph.
/// Complexity: O(n + m)
pub fn hyper_degeneracy_ordering<W>(adj: &HyperAdjList<W>) -> (OrderAndPos, usize) {
    let n = adj.n();
    if n == 0 {
        return (OrderAndPos::empty(), 0);
    }

    let mut deg = adj
        .adj
        .iter()
        .map(|neighbors| neighbors.len())
        .collect::<Vec<_>>();
    let max_deg = *deg.iter().max().unwrap_or(&0);

    let mut bin_count = vec![0; max_deg + 1];
    for &d in &deg {
        bin_count[d] += 1;
    }

    let mut bin_starts = vec![0; max_deg + 1];
    let mut start_pos = 0;
    for d in 0..=max_deg {
        bin_starts[d] = start_pos;
        start_pos += bin_count[d];
    }

    let mut temp_starts = bin_starts.clone();
    let mut order = vec![0; n];
    let mut pos = vec![0; n];
    for v in 0..n {
        pos[v] = temp_starts[deg[v]];
        order[pos[v]] = v;
        temp_starts[deg[v]] += 1;
    }

    let mut peeled = vec![false; adj.m()];
    let mut k = 0;
    for i in 0..n {
        let v = order[i];
        k = std::cmp::max(k, deg[v]);

        for (edge_id, edge) in adj.iter_incident_edges(v as NodeId) {
            if peeled[edge_id as usize] {
                continue;
            }
            peeled[edge_id as usize] = true;

            for &n_node in edge.nodes {
                let u = n_node as usize;
                let u_deg = deg[u];
                let u_pos = pos[u];

                let first_node_pos = bin_starts[u_deg];
                let first_node = order[first_node_pos];

                if u != first_node {
                    pos.swap(u, first_node);
                    order.swap(u_pos, first_node_pos);
                }

                bin_starts[u_deg] += 1;
                deg[u] -= 1;
            }
        }
    }

    (
        OrderAndPos::new(order.into_iter().map(|e| e as NodeId).collect(), pos, true),
        k,
    )
}

pub type Order = Vec<NodeId>;

pub type Pos = Vec<usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderAndPos {
    pub order: Vec<NodeId>,
    pub pos: Vec<usize>,
    pub ascending: bool,
}

impl OrderAndPos {
    pub fn new(order: Vec<NodeId>, pos: Vec<usize>, ascending: bool) -> Self {
        Self {
            order,
            pos,
            ascending,
        }
    }

    pub fn empty() -> Self {
        Self {
            order: Vec::new(),
            pos: Vec::new(),
            ascending: true,
        }
    }

    pub fn reverse(&mut self) {
        self.order.reverse();
        for (p, v) in self.order.iter().enumerate() {
            self.pos[*v as usize] = p;
        }
        self.ascending = !self.ascending;
    }

    pub fn take_order(&mut self) -> Order {
        std::mem::take(&mut self.order)
    }

    pub fn take_pos(&mut self) -> Pos {
        std::mem::take(&mut self.pos)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderOrPos {
    Order(Order),
    Pos(Pos),
}

impl OrderOrPos {
    pub fn get_order(&self) -> Cow<'_, [NodeId]> {
        match self {
            OrderOrPos::Order(order) => Cow::Borrowed(order),
            OrderOrPos::Pos(pos) => {
                let mut rv = vec![0; pos.len()];
                for (i, &p) in pos.iter().enumerate() {
                    rv[p] = i as NodeId;
                }
                Cow::Owned(rv)
            }
        }
    }
    pub fn get_pos(&self) -> Cow<'_, [usize]> {
        match self {
            OrderOrPos::Order(order) => {
                let mut rv = vec![0; order.len()];
                for (i, &node) in order.iter().enumerate() {
                    rv[node as usize] = i;
                }
                Cow::Owned(rv)
            }
            OrderOrPos::Pos(pos) => Cow::Borrowed(pos),
        }
    }
}

// pub enum OrderRef<'a> {
//     Order(&'a [NodeId]),
//     Pos(&'a [usize]),
// }
//
// impl<'a> OrderRef<'a> {
//     pub fn get_order(&'a self) -> Cow<'a, [NodeId]> {
//         match self {
//             OrderRef::Order(order) => Cow::Borrowed(order),
//             OrderRef::Pos(pos) => {
//                 let mut rv = vec![0; pos.len()];
//                 for (i, &p) in pos.iter().enumerate() {
//                     rv[p] = i as NodeId;
//                 }
//                 Cow::Owned(rv)
//             }
//         }
//     }
//     pub fn get_pos(&'a self) -> Cow<'a, [usize]> {
//         match self {
//             OrderRef::Order(order) => {
//                 let mut rv = vec![0; order.len()];
//                 for (i, &node) in order.iter().enumerate() {
//                     rv[node as usize] = i;
//                 }
//                 Cow::Owned(rv)
//             }
//             OrderRef::Pos(pos) => Cow::Borrowed(pos),
//         }
//     }
// }
