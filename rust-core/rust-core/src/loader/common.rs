use super::error::*;
use std::fs::{self, File, create_dir_all};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use std::collections::HashMap;

use super::DatasetLoaderDispatcherAttr;

use crate::misc::serialize::{DumpCacheToFile, LoadFromCacheDeserialized};
use crate::types::{
    Hypergraph, NodeId, NodeWeight, UnweightedHx, UnweightedHypergraph, WeightedHx,
    WeightedHypergraph,
};
use serde::Deserialize;

/// Struct to hold the dataset information specified in dataset.toml
#[derive(Deserialize, Debug)]
pub struct DatasetConfig {
    pub cache_dir: Option<PathBuf>,
    #[serde(flatten)]
    pub datasets: HashMap<String, DatasetDescriptor>,
}

#[derive(Deserialize, Debug)]
pub struct DatasetDescriptor {
    pub path: PathBuf,
    pub alias: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub description: Option<String>,
}

pub fn parse_datasets_descriptor() -> Result<DatasetConfig, Box<dyn std::error::Error>> {
    let path_str = std::env::var("DATASETS_TOML")?;
    let toml_str = fs::read_to_string(path_str)?;
    let config: DatasetConfig = toml::from_str(&toml_str)?;
    Ok(config)
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
    /// The name of the dataset, used for logging and cache file naming.
    const NAME: &'static str;
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

pub fn hash_to_len<T: Hash>(x: T, length: usize) -> String {
    let mut hasher = DefaultHasher::new();
    x.hash(&mut hasher);
    let hash_val = hasher.finish();

    let hash_string = format!("{:016x}", hash_val);

    if length >= hash_string.len() {
        format!("{:0>width$}", hash_string, width = length)
    } else {
        hash_string[..length].to_string()
    }
}

pub trait Loader
where
    Self: DatasetLoaderDispatcherAttr + Hash,
    Self::Output: DumpCacheToFile + LoadFromCacheDeserialized,
{
    /// A description of the method used to load the dataset, for caching purposes
    const VARIANT: &'static str;

    type Output;

    fn load(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.get_dataset_location();
        let cache_dir = self.get_cache_dir();

        if !dataset_location.exists() {
            log::error!("Invalid dataset location '{}'", dataset_location.display());
            return Err(LoaderError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Invalid dataset location",
            )));
        }
        if cache_dir.is_none() {
            return self.from_file();
        }

        if !*self.get_cached() {
            return self.from_file();
        }

        let cache_dir = cache_dir.as_ref().unwrap();

        let cache_file = PathBuf::from(&cache_dir)
            .join(format!(
                "{}_{}_{}",
                self.get_name(),
                Self::VARIANT,
                hash_to_len(self, 16)
            ))
            .with_extension("bin");

        if cache_file.exists() {
            match <Self::Output as LoadFromCacheDeserialized>::load_deserialized(&cache_file) {
                Ok(hg) => Ok(hg),
                Err(_err) => {
                    log::warn!(
                        "Cache file {} is corrupted. Falling back to uncached loading.",
                        cache_file.display()
                    );
                    let rv = self.from_file()?;
                    if let Err(e) = rv.save_to_file(&cache_file) {
                        log::error!(
                            "Failed to save hypergraph to cache file {}: {}",
                            cache_file.display(),
                            e
                        );
                    }
                    Ok(rv)
                }
            }
        } else {
            log::info!(
                "Loading hypergraph from source and caching to {}...",
                cache_file.display()
            );
            let rv = self.from_file()?;

            if let Err(e) = rv.save_to_file(&cache_file) {
                log::error!(
                    "Failed to save hypergraph to cache file {}: {}",
                    cache_file.display(),
                    e
                );
            }
            Ok(rv)
        }
    }

    fn from_file(&self) -> Result<Self::Output, LoaderError>;
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
