use duplicate::duplicate_item;

/// Represents a motifs of size 2 to 6 in a memory efficient way
///
/// ith bit is 1 if the corresponding edge is present
///
/// edges are stored in ascending order of size (order), i.e. first all size 2 edges, then all size
/// 3 edges, etc.
/// inside the order, edges are stored in lexicographical order. For example, in the order 3 chunk
/// the first edge will be (0,1,2), then (0,1,3); last one will be (2,3,4)
struct CompressedMotif<const N: usize, T> {
    rep: T,
    //10 size 2
    //10 size 3
    //5  size 4
    //1  size 5
}

#[duplicate_item(
    f_name                      e_bitset    n_bitset   max_hx_size;
    [compute_constants_u32]     [ u32 ]     [ u8 ]     [4];
    [compute_constants_u64]     [ u64 ]     [ u8 ]     [6];
)]
pub const fn f_name<const N: usize>() -> ([e_bitset; 64], [n_bitset; 64]) {
    // v[i] = all edges touched by the i hyperedge
    let mut overlaps = [0 as e_bitset; 64];
    // V[i] = all nodes in the i hyperedge
    let mut nodes = [0 as n_bitset; 64];

    // 2-edges
    let mut pivot_start = 0;
    let offset = N - 1;

    let mut i = 0;
    while i < N {
        nodes[i] |= (1 << (N - i) - 1) << (pivot_start);
        pivot_start += N - i;
        i += 1;
    }

    // let x: e_bitset = 14;
    (overlaps, nodes)
}

fn test() {
    let x: u32 = 12;
}

// impl<const N: usize, T> CompressedMotif<N, T> {
//     // let x: u32 = 12;
//     // const fn get_
// }
