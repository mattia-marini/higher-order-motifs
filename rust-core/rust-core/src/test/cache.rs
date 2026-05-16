use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use crate::loader::load_conference;

#[test]
pub fn big() {
    // let time = Instant::now();
    // let dataset_dir = std::env::var("DATASET_DIR").unwrap();
    // let cache_dir = std::env::var("CACHE_DIR").unwrap();
    //
    // match load_wiki_talk(&dataset_dir, Some(&cache_dir)) {
    //     Ok(hg) => println!("edges.len m {}", hg.m()),
    //     Err(e) => assert!(false, "Failed to load dataset: {}", e),
    // }
    // println!("Total time: {:?}", time.elapsed());
}

#[test]
pub fn small() {
    let time = Instant::now();
    let dataset_dir = std::env::var("DATASET_DIR").unwrap();
    let dataset_path = PathBuf::from(dataset_dir).join("conference.dat");

    println!("{}", dataset_path.display());

    let cache_dir = std::env::var("CACHE_DIR").unwrap();

    match load_conference(&dataset_path, &cache_dir) {
        Ok(hg) => println!("edges.len m {}", hg.m()),
        Err(e) => assert!(false, "Failed to load dataset: {}", e),
    }
    println!("Total time: {:?}", time.elapsed());
}
