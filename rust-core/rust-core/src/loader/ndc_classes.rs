use std::error::Error;
use std::fs::read_to_string;
use std::path::Path;

use seq_macro::seq;

use crate::types::{
    Hx, Hypergraph, NodeId, NodeWeight, UnweightedHx, UnweightedHypergraph, WeightedHx,
    WeightedHypergraph,
};
use crate::{loader::common::Loader, loader::error::LoaderError};

use super::{NdcClassesStdUnweightedLoader, NdcClassesStdWeightedLoader};

pub struct Unweighted;
pub struct Weighted;

fn read_ints_from_file<P: AsRef<Path>>(path: &P) -> Result<Vec<NodeId>, LoaderError> {
    let s = read_to_string(path)?;
    let mut v = Vec::new();
    for line in s.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Ok(x) = t.parse::<NodeId>() {
            v.push(x);
        }
    }
    Ok(v)
}

impl Loader for NdcClassesStdUnweightedLoader {
    type Output = UnweightedHypergraph;

    const VARIANT: &'static str = "uw";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        let base = dataset_location;
        let nverts_path = if base.is_dir() {
            base.join(format!(
                "{}-nverts.txt",
                base.file_name().and_then(|n| n.to_str()).unwrap_or("")
            ))
        } else {
            base.with_extension("-nverts.txt")
        };
        let simplices_path = if base.is_dir() {
            base.join(format!(
                "{}-simplices.txt",
                base.file_name().and_then(|n| n.to_str()).unwrap_or("")
            ))
        } else {
            base.with_extension("-simplices.txt")
        };

        let v = read_ints_from_file(&nverts_path)?;
        let mut s = read_ints_from_file(&simplices_path)?;

        let mut hg = Hypergraph::new();

        for mut i in v.into_iter() {
            let mut e: Vec<NodeId> = Vec::new();
            for _ in 0..i {
                if s.is_empty() {
                    break;
                }
                e.push(s.remove(0));
            }
            if e.len() > 1 && e.len() <= 10 {
                seq!(N in 2..11 {
                    if e.len() == N {
                        let mut arr = [0 as NodeId; N];
                        for j in 0..N { arr[j] = e[j]; }
                        hg.add_edge(Hx::new(arr, ()).expect("Malformed edge"));
                    }
                });
            }
        }

        Ok(hg.into())
    }
}

impl Loader for NdcClassesStdWeightedLoader {
    type Output = WeightedHypergraph;

    const VARIANT: &'static str = "w";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        let base = dataset_location;
        let nverts_path = if base.is_dir() {
            base.join(format!(
                "{}-nverts.txt",
                base.file_name().and_then(|n| n.to_str()).unwrap_or("")
            ))
        } else {
            base.with_extension("-nverts.txt")
        };
        let simplices_path = if base.is_dir() {
            base.join(format!(
                "{}-simplices.txt",
                base.file_name().and_then(|n| n.to_str()).unwrap_or("")
            ))
        } else {
            base.with_extension("-simplices.txt")
        };

        let v = read_ints_from_file(&nverts_path)?;
        let mut s = read_ints_from_file(&simplices_path)?;

        let mut hg = Hypergraph::new();

        for mut i in v.into_iter() {
            let mut e: Vec<NodeId> = Vec::new();
            for _ in 0..i {
                if s.is_empty() {
                    break;
                }
                e.push(s.remove(0));
            }
            if e.len() > 1 && e.len() <= 10 {
                seq!(N in 2..11 {
                    if e.len() == N {
                        let mut arr = [0 as NodeId; N];
                        for j in 0..N { arr[j] = e[j]; }
                        if !hg.has_hyperedge(&arr) {
                            hg.add_edge(Hx::new(arr, 0.0).expect("Malformed edge")); }
                        hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                    }
                });
            }
        }

        Ok(hg.into())
    }
}
