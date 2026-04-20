use std::time::Instant;

use crate::loader::load_wiki_talk;

#[test]
pub fn big() {
    let time = Instant::now();
    let dataset_dir = std::env::var("DATASET_DIR").unwrap();
    let cache_dir = std::env::var("CACHE_DIR").unwrap();

    if let Err(e) = load_wiki_talk(&dataset_dir, &cache_dir) {
        assert!(false, "Failed to load dataset: {}", e);
    }
    println!("Loaded in {:?}", time.elapsed());
}
