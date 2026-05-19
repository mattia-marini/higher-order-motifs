use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::graph::{AdjList, Hypergraph, NodeId, UnweightedHypergraph};
use crate::loader::common::{Loader, get_dataset_paths, parse_u32};

const PATH: &str = "wiki/wiki-talk.txt";

pub struct Unweighted2Uniform {}

impl Loader for Unweighted2Uniform {
    const NAME: &'static str = "UW_primary_school";

    type Output = AdjList;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
        let file = File::open(dataset_location)?;
        let reader = BufReader::new(file);
        let mut edges: Vec<(NodeId, NodeId)> = Vec::with_capacity(1_000_000);

        for line in reader.lines() {
            let l = line?;
            let parts: Vec<&str> = l.split_whitespace().collect();
            if parts.len() == 2 {
                let u = parts[0].parse::<NodeId>()?;
                let v = parts[1].parse::<NodeId>()?;
                edges.push((u, v));
            }
        }

        Ok(AdjList::from_edges_mapped(edges).0)
    }
}
