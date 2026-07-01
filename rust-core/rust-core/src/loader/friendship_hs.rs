use crate::types::{
    Hypergraph, NodeId, NodeWeight, UnweightedHx, UnweightedHypergraph, WeightedHx,
    WeightedHypergraph,
};
use crate::{loader::common::Loader, loader::error::LoaderError};
use std::{error::Error, path::Path};

use super::{FriendshipHsStdUnweightedLoader, FriendshipHsStdWeightedLoader};

pub struct Unweighted;
pub struct Weighted;

impl Loader for FriendshipHsStdUnweightedLoader {
    type Output = UnweightedHypergraph;

    const VARIANT: &'static str = "uw";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
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

        Ok(hg.into())
    }
}

impl Loader for FriendshipHsStdWeightedLoader {
    type Output = WeightedHypergraph;

    const VARIANT: &'static str = "w";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
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
                    hg.add_edge(WeightedHx::new_unchecked([a, b], 0.0).0);
                }
                hg.modify_hx_weigth_with::<2, _>(&[a, b], |w| w + 1.0);
            }
        }

        Ok(hg.into())
    }
}
