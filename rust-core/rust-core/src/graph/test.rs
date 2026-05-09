// use crate::graph::Hx;
// pub struct CTMap<const N: usize> {
//     buckets: [Vec<Hx>; N],
// }

// pub struct CTMap<const N: usize> {
//     // buckets: [Vec<Hx>; N],
// }
//
// impl<const N: usize> CTMap<N> {
//     pub fn new() -> Self {
//         // let buckets = [(); N].map(|_| Vec::new());
//         Self {
//             // buckets,
//         }
//     }
//
//     pub fn get<const M: usize>(&self) {
//         const { assert!(M < N, "Index out of bounds at compile time!") };
//         // if M >= N {
//         //     panic!("Index out of bounds");
//         // }
//         unimplemented!()
//     }
// }

// pub struct CTMap<const N: usize>;
//
// impl<const N: usize> CTMap<N> {
//     pub fn new() -> Self {
//         Self
//     }
//
//     pub fn get<const I: usize>(&self) {
//         // This constant will only compile if the condition is true.
//         const ASSERT: () = ();
//         let _ = BoundCheck::<I, N>::VALID;
//
//         println!("Accessing index {}", I);
//     }
// }
//
// struct BoundCheck<const I: usize, const N: usize>;
//
// impl<const I: usize, const N: usize> BoundCheck<I, N> {
//     const VALID: () = {
//         if I >= N {
//             // In a const context, a panic causes a compile-time error.
//             panic!("Index out of bounds!");
//         }
//     };
// }

// pub trait Bucket<const N: usize> {
//     type BucketType;
//     fn get(&self) -> &Self::BucketType;
//     fn get_mut(&mut self) -> &mut Self::BucketType;
// }
//
// impl<const N: usize> Bucket<N> for CTMap {
//     type BucketType = usize;
//
//     fn get(&self) -> &Self::BucketType {
//         &self.buckets[N]
//     }
//
//     fn get_mut(&mut self) -> &mut Self::BucketType {
//         &mut self.buckets[N]
//     }
//

use rust_core_macros::define_ct_map;

define_ct_map!(2..10, [NodeId; N]);

pub fn test() {
    // let x = Map2_10::new();
    // let x = CTMap::<4>::new();
    // x.get::<2>();
    // x.get::<5>();
}
