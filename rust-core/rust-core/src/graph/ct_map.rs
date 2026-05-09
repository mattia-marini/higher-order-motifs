use rust_core_macros::ct_map;

#[ct_map(ty(Vec::<[u32; N]>), rg(2..5))]
struct CTMap;

// fn test() {
//     let mut map = CTMap::new();
//     // let x = map.get_4_mut();
//     // let x = map.take_4();
//     let x = map.take_3();
// }
