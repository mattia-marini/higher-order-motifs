use std::error::Error;
use std::fs::{self, File, create_dir_all};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::graph::Hypergraph;
use crate::graph::serialize::{DumpCacheToFile, LoadFromCacheDeserialized};

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

fn hash_file_metadata<P: AsRef<Path>>(path: P) -> io::Result<u64> {
    let metadata = fs::metadata(path)?;
    let size = metadata.len();
    let mtime = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let mut hasher = DefaultHasher::new();

    size.hash(&mut hasher);
    mtime.hash(&mut hasher);

    Ok(hasher.finish())
}

pub fn get_cache_file<P1, P2>(
    dataset_location: &P1,
    cache_dir: &P2,
    name: &str,
    extension: &str,
) -> io::Result<PathBuf>
where
    P1: AsRef<Path> + ?Sized,
    P2: AsRef<Path> + ?Sized,
{
    let hash = hash_file_metadata(dataset_location)?;
    Ok(PathBuf::from(cache_dir.as_ref())
        .join(format!("{}_{:016x}", name, hash))
        .with_extension(extension))
}

pub trait DatasetInfo {
    /// The folder or file path where the raw dataset is located.
    /// loader will always read from the raw dataset file.
    fn dataset_location(&self) -> PathBuf;
    /// The directory where the cache file should be stored. If `None`, caching is disabled and the
    /// loader will always read from the raw dataset file.
    ///This parameter is set from the dataset.toml file
    fn cache_dir(&self) -> Option<PathBuf>;
    /// Returns a string that should uniquely identify the dataset. Implementation is generated
    /// through the `#[loader]` macro, and is used to determine the cache file name.
    fn cache_hash(&self, length: usize) -> String;
}

pub trait Loader
where
    Self: DatasetInfo,
    Self::Output: DumpCacheToFile + LoadFromCacheDeserialized,
{
    // const DATASET_PATH: &'static str;
    // const CACHE_DIR: &'static str;
    // const NAME: &'static str;
    type Output;

    fn load(&self) -> Result<Self::Output, Box<dyn Error>> {
        let dataset_location = self.dataset_location();
        let cache_dir = self.cache_dir();

        if !dataset_location.exists() {
            log::error!("Invalid dataset location '{}'", dataset_location.display());
            return Err(Box::new(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "Dataset file '{}' does not exist",
                    dataset_location.display()
                ),
            )));
        }
        if cache_dir.is_none() {
            return self.from_file();
        }
        let cache_dir = cache_dir.unwrap();

        match get_cache_file(&dataset_location, &cache_dir, &self.cache_hash(16), "bin") {
            Ok(cache_file) => {
                if cache_file.exists() {
                    match <Self::Output as LoadFromCacheDeserialized>::load_deserialized(
                        &cache_file,
                    ) {
                        Ok(hg) => Ok(hg),
                        Err(_err) => {
                            log::warn!(
                                "Cache file {} is corrupted. Falling back to uncached loading.",
                                cache_file.display()
                            );
                            let rv = self.from_file()?;
                            rv.save_to_file(&cache_file)?;
                            Ok(rv)
                        }
                    }
                } else {
                    log::info!(
                        "Loading hypergraph from source and caching to {}...",
                        cache_file.display()
                    );
                    let rv = self.from_file()?;
                    rv.save_to_file(&cache_file)?;
                    Ok(rv)
                }
            }
            Err(err) => {
                log::warn!(
                    "Cannot hash input file '{}': {}. Falling back to uncached loading.",
                    dataset_location.display(),
                    err
                );
                Ok(self.from_file()?)
            }
        }
    }

    fn from_file(&self) -> Result<Self::Output, Box<dyn Error>>;

    // fn cache_file_name() -> &'static str {
    //     // Name constants were removed; use DatasetInfo::cache_hash instead to obtain
    //     // a stable identifier for cache files.
    //     "<dataset-name>"
    // }
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
