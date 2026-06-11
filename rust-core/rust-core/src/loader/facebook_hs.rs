use crate::{
    graph::{Hypergraph, NodeId, UnweightedHypergraph},
    loader::common::Loader,
};
use std::{error::Error, path::Path};

use super::FacebookHsStdUnweightedLoader;

impl Loader for FacebookHsStdUnweightedLoader {
    type Output = UnweightedHypergraph;

    fn from_file(&self) -> Result<Self::Output, Box<dyn Error>> {
        let dataset_location = self.dataset_location.clone();
        // The python loader uses pandas and only reads triples (a,b,c) and if only_confirmed then c==1.
        // Here we'll load the file naively, assuming whitespace-separated columns a b c per line.
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(dataset_location)?;
        let reader = BufReader::new(file);

        let mut hg = Hypergraph::new();

        for line in reader.lines() {
            let l = line?;
            let parts: Vec<&str> = l.split_whitespace().collect();
            if parts.len() >= 2 {
                let a = parts[0].parse().unwrap_or(0);
                let b = parts[1].parse().unwrap_or(0);
                // Only include confirmed if a third column exists and equals 1, otherwise include by default
                let include = if parts.len() >= 3 {
                    parts[2] == "1" || parts[2] == "1.0"
                } else {
                    true
                };
                if include {
                    hg.add_edge(crate::graph::UnweightedHx::new_unchecked([a, b]).0);
                }
            }
        }

        Ok(hg.into())
    }
}
