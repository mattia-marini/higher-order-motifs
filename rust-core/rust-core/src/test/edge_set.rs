use hashbrown::HashSet;

use crate::graph::{
    edge_collection::{StaticEdgeSet, StaticEdgeSetAccessor},
    types2::Hx,
};

#[test]
pub fn test() {
    let mut x = StaticEdgeSet::new();
    let y = Hx::new_unweighted_unchecked([1, 2]);

    x.insert(Hx::new_unchecked([1, 2, 3, 4], 3));
    x.insert(Hx::new_unchecked([1, 2, 3, 4], 2));
    x.insert(Hx::new_unchecked([1, 2, 3, 4, 5], 2));
    println!("{:?}", x);
}
