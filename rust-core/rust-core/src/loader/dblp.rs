use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use hashbrown::HashMap;
use seq_macro::seq;

use crate::{
    graph::{Hx, Hypergraph, NodeId, NodeWeight},
    loader::common::Loader,
};

use super::{DblpStdUnweightedLoader, DblpStdWeightedLoader};

impl Loader for DblpStdUnweightedLoader {
    type Output = crate::graph::UnweightedHypergraph;

    fn from_file(&self) -> Result<Self::Output, Box<dyn Error>> {
        let dataset_location = self.dataset_location.clone();
        let file = File::open(dataset_location)?;
        let reader = BufReader::new(file);

        let mut graph: HashMap<String, Vec<NodeId>> = HashMap::new();
        for line in reader.lines().skip(1) {
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
            a.sort_unstable();
            a.dedup();
            if a.len() > 1 && a.len() <= 10 {
                seq!(N in 2..11 {
                    if a.len() == N {
                        let mut arr = [0 as NodeId; N];
                        for i in 0..N { arr[i] = a[i]; }
                        hg.add_edge(Hx::new_unchecked(arr, ()));
                    }
                });
            }
        }

        Ok(hg.into())
    }
}

impl Loader for DblpStdWeightedLoader {
    type Output = crate::graph::WeightedHypergraph;

    fn from_file(&self) -> Result<Self::Output, Box<dyn Error>> {
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
            a.sort_unstable();
            a.dedup();

            if a.len() > 1 && a.len() <= 10 {
                seq!(N in 2..11 {
                    if a.len() == N {
                        let mut arr = [0 as NodeId; N];
                        for i in 0..N { arr[i] = a[i]; }
                        if !hg.has_hyperedge(&arr) { hg.add_edge(Hx::new_unchecked(arr, 0.0)); }
                        hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                    }
                });
            }
        }

        Ok(hg.into())
    }
}
