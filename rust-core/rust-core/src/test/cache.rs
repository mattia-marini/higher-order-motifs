use std::time::Instant;

use crate::{graph::ArchivedUnweightedHypergraph, loader::load_wiki_talk};

#[test]
pub fn big() {
    let time = Instant::now();
    let dataset_dir = std::env::var("DATASET_DIR").unwrap();
    let cache_dir = std::env::var("CACHE_DIR").unwrap();

    match load_wiki_talk(&dataset_dir, Some(&cache_dir)) {
        Ok(hg) => println!("edges.len m {}", hg.m()),
        Err(e) => assert!(false, "Failed to load dataset: {}", e),
    }
    println!("Total time: {:?}", time.elapsed());
}
