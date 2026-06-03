use std::error::Error;
use std::fs::read_to_string;
use std::path::Path;

use seq_macro::seq;

use crate::{
    graph::{Hx, Hypergraph, NodeId, NodeWeight},
    loader::common::Loader,
};

pub struct Unweighted;
pub struct Weighted;

fn read_ints_from_file<P: AsRef<Path>>(path: &P) -> Result<Vec<NodeId>, Box<dyn Error>> {
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

impl Loader for Unweighted {
    const NAME: &'static str = "UW_enron";
    type Output = Hypergraph<NodeId, ()>;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
        let base = dataset_location.as_ref();
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
                        hg.add_edge(Hx::new(arr, ()).expect(format!("[{}] Malformed edge", Self::NAME).as_str()));
                    }
                });
            }
        }

        Ok(hg)
    }
}

impl Loader for Weighted {
    const NAME: &'static str = "W_enron";
    type Output = Hypergraph<NodeId, NodeWeight>;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
        let base = dataset_location.as_ref();
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
                        if !hg.has_hyperedge(&arr) { hg.add_edge(Hx::new(arr, 0.0).expect(format!("[{}] Malformed edge", Self::NAME).as_str())); }
                        hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                    }
                });
            }
        }

        Ok(hg)
    }
}
