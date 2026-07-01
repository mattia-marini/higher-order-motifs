use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use hashbrown::HashMap;
use seq_macro::seq;

use crate::types::{
    Hx, Hypergraph, NodeId, NodeWeight, UnweightedHx, UnweightedHypergraph, WeightedHx, WeightedHypergraph
};
use crate::{loader::common::Loader, loader::error::LoaderError};

use super::{GeologyStdUnweightedLoader, GeologyStdWeightedLoader};

pub struct Unweighted;
pub struct Weighted;

impl Loader for GeologyStdUnweightedLoader {
    type Output = UnweightedHypergraph;

    const VARIANT: &'static str = "uw";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        let file = File::open(dataset_location)?;
        let reader = BufReader::new(file);

        let mut graph: HashMap<String, Vec<NodeId>> = HashMap::new();

        for line in reader.lines() {
            let l = line?;
            if l.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() >= 2 {
                let paper = parts[0].to_string();
                if let Ok(author) = parts[1].trim().parse::<NodeId>() {
                    graph.entry(paper).or_insert_with(Vec::new).push(author);
                }
            }
        }

        let mut hg = Hypergraph::new();

        for (_paper, authors) in graph.into_iter() {
            let mut a = authors;
            if a.len() > 1 && a.len() <= 10 {
                seq!(N in 2..11 {
                    if a.len() == N {
                        let mut arr = [0 as NodeId; N];
                        for i in 0..N { arr[i] = a[i]; }
                        hg.add_edge(Hx::new(arr, ()).expect("Malformed edge"));
                    }
                });
            }
        }

        Ok(hg.into())
    }
}

impl Loader for GeologyStdWeightedLoader {
    type Output = WeightedHypergraph;

    const VARIANT: &'static str = "w";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        let file = File::open(dataset_location)?;
        let reader = BufReader::new(file);

        let mut graph: HashMap<String, Vec<NodeId>> = HashMap::new();

        for line in reader.lines() {
            let l = line?;
            if l.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() >= 2 {
                let paper = parts[0].to_string();
                if let Ok(author) = parts[1].trim().parse::<NodeId>() {
                    graph.entry(paper).or_insert_with(Vec::new).push(author);
                }
            }
        }

        let mut hg = Hypergraph::new();

        for (_paper, authors) in graph.into_iter() {
            let mut a = authors;
            if a.len() > 1 && a.len() <= 10 {
                seq!(N in 2..11 {
                    if a.len() == N {
                        let mut arr = [0 as NodeId; N];
                        for i in 0..N { arr[i] = a[i]; }
                        if !hg.has_hyperedge(&arr) { hg.add_edge(Hx::new(arr, 0.0).expect("Malformed edge")); }
                        hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                    }
                });
            }
        }

        Ok(hg.into())
    }
}
