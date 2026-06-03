use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use crate::loader::{common::Loader, conference, pacs::load_pacs_common_from_csv};

// #[test]
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

// #[test]
pub fn small() {
    let time = Instant::now();
    let dataset_dir = std::env::var("DATASET_DIR").unwrap();
    let dataset_location = PathBuf::from(dataset_dir).join("conference.dat");

    // println!("{}", dataset_location.display());

    let cache_dir = std::env::var("CACHE_DIR").unwrap();

    match conference::Unweighted::load(&dataset_location, &cache_dir) {
        Ok(hg) => println!("edges.len m {}", hg.m()),
        Err(e) => assert!(false, "Failed to load dataset: {}", e),
    }
    // println!("Total time: {:?}", time.elapsed());
}

// #[test]
pub fn polars() {
    let dataset_dir = std::env::var("DATASET_DIR").unwrap();
    let pacs_location = PathBuf::from(dataset_dir).join("PACS.csv");
    load_pacs_common_from_csv(&pacs_location).unwrap();
}
