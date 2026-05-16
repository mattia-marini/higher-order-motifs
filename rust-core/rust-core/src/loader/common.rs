use std::error::Error;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};

/// Returns the paths for the dataset and cache files based on the provided relative path. Creates
/// the cache directory if it does not exist.
pub fn get_dataset_paths<P1, P2, P3>(
    dataset_dir: &P1,
    cache_dir: &P2,
    path: &P3,
) -> Result<(PathBuf, PathBuf), Box<dyn Error>>
where
    P1: AsRef<Path> + ?Sized,
    P2: AsRef<Path> + ?Sized,
    P3: AsRef<Path> + ?Sized,
{
    // let dataset_dir = std::env::var("DATASET_DIR")?;
    // let cache_dir = std::env::var("CACHE_DIR")?;

    let dataset_path = dataset_dir.as_ref().join(path.as_ref());
    let cache_path = cache_dir.as_ref().join(path.as_ref().with_extension("bin"));

    create_dir_all(cache_path.parent().unwrap())?;

    Ok((dataset_path, cache_path))
}

#[inline(always)]
pub fn parse_u32(chars: &[u8]) -> u32 {
    let mut rv = 0;
    let mut base = 1;
    for c in chars.iter().rev() {
        rv += (c - b'0') as u32 * base;
        base *= 10;
    }
    rv
}
