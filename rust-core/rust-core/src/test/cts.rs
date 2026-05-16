use crate::graph::edge_collection::{CtHxSetSizedAccessor, StaticEdgeSet};

#[test]
pub fn test() {
    let mut x = StaticEdgeSet::<i32, i32>::new();
    let bucket: &HashSet<Hx<0, i32, i32>> = x.get_bucket();
}
