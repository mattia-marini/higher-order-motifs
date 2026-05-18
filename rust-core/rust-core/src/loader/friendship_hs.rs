use crate::{
    graph::{Hypergraph, NodeId, NodeWeight, UnweightedHx},
    loader::common::Loader,
};
use std::{error::Error, path::Path};

pub struct Unweighted;
pub struct Weighted;

impl Loader for Unweighted {
    const NAME: &'static str = "UW_friendship_hs";
    type Output = Hypergraph<NodeId, ()>;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
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
                hg.add_edge(UnweightedHx::new_unchecked([a, b]).0);
            }
        }

        Ok(hg)
    }
}

impl Loader for Weighted {
    const NAME: &'static str = "W_friendship_hs";
    type Output = Hypergraph<NodeId, NodeWeight>;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
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
                if !hg.has_hyperedge::<2>(&[a, b]) {
                    hg.add_edge(crate::graph::WeightedHx::new_unchecked([a, b], 0.0).0);
                }
                hg.modify_hx_weigth_with::<2, _>(&[a, b], |w| w + 1.0);
            }
        }

        Ok(hg)
    }
}
