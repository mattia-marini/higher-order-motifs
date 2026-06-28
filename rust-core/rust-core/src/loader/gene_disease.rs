use hashbrown::HashMap;
use seq_macro::seq;
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::{
    graph::{Hx, Hypergraph, NodeId, NodeWeight, WeightedHypergraph},
    loader::common::Loader,
    loader::error::LoaderError,
};

use super::GeneDiseaseStdWeightedLoader;

pub struct Weighted;

impl Loader for GeneDiseaseStdWeightedLoader {
    type Output = WeightedHypergraph;

    const VARIANT: &'static str = "w";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        // Parse TSV and aggregate diseases -> list of genes
        let file = File::open(dataset_location)?;
        let mut reader = BufReader::new(file);

        let mut diseases: HashMap<String, Vec<NodeId>> = HashMap::new();

        let mut line = String::new();
        while reader.read_line(&mut line)? > 0 {
            let s = line.trim();
            if s.is_empty() {
                line.clear();
                continue;
            }
            let parts: Vec<&str> = s.split('\t').collect();
            if parts.len() > 4 {
                if let Ok(gene) = parts[0].parse::<NodeId>() {
                    let dis = parts[4].to_string();
                    diseases.entry(dis).or_insert_with(Vec::new).push(gene);
                }
            }
            line.clear();
        }

        let mut hg = Hypergraph::new();

        for (_d, mut genes) in diseases.into_iter() {
            if genes.len() > 1 {
                // keep only up to size 10 to match other loaders
                if genes.len() <= 10 {
                    seq!(N in 2..11 {
                        if genes.len() == N {
                            let mut arr = [0 as NodeId; N];
                            for i in 0..N { arr[i] = genes[i]; }
                            if !hg.has_hyperedge(&arr) {
                                hg.add_edge(Hx::new_unchecked(arr, 0.0));
                            }
                            hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                        }
                    });
                }
            }
        }

        Ok(hg.into())
    }
}
