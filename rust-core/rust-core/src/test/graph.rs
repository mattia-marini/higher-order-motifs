use crate::graph::{CtHxVecAccessor, Hx, NodeId};
pub trait X<const N: usize> {
    fn get(&self) -> &[u32; N];
}

#[test]
pub fn ct_map() {
    use crate::graph::unweighted_hypergraph::CtHxVec;
    let mut map = CtHxVec::new();

    map.push(Hx::new_unchecked([1, 2, 3]));
    map.push(Hx::new_unchecked([1, 2]));
    let rv: &mut Vec<Hx<2, NodeId>> = map.get_mut();
    // let rv = <map as CtHxVecAccessor<2>>.get_mut();
    // let rv = map.get_mut();

    println!("map: {:?}", map);
}
