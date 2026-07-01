use std::error::Error;
use std::fs::read_to_string;
use std::path::Path;

use hashbrown::HashMap;
use seq_macro::seq;

use crate::types::{
    Hx, Hypergraph, NodeId, NodeWeight, UnweightedHx, UnweightedHypergraph, WeightedHx,
    WeightedHypergraph,
};
use crate::{loader::common::Loader, loader::error::LoaderError};

pub struct Unweighted;
pub struct Weighted;

use super::{WikiStdUnweightedLoader, WikiStdWeightedLoader};

impl Loader for WikiStdUnweightedLoader {
    type Output = UnweightedHypergraph;

    const VARIANT: &'static str = "uw";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        let contents = read_to_string(dataset_location)?;
        let mut votes: HashMap<String, Vec<String>> = HashMap::new();
        let mut hg = Hypergraph::new();

        for line in contents.lines() {
            let l = line.trim();
            if l.is_empty() {
                // flush votes
                for (_k, v) in votes.drain() {
                    let uids: Vec<NodeId> = v
                        .into_iter()
                        .filter_map(|s| s.parse::<NodeId>().ok())
                        .collect();
                    let order = uids.len();
                    seq!(N in 2..11 {
                        match order {
                            N => {
                                let mut arr = [0 as NodeId; N];
                                arr.copy_from_slice(&uids);
                                hg.add_edge(Hx::new(arr, ()).expect("wiki: found malformed hyperedge"));
                            },
                            _ => ()
                        }
                    });
                }
                continue;
            }
            if !l.starts_with('V') {
                continue;
            }
            let parts: Vec<&str> = l.split_whitespace().collect();
            if parts.len() < 3 {
                continue;
            }
            let vote = parts[1].to_string();
            let u_id = parts[2].to_string();
            votes.entry(vote).or_insert_with(Vec::new).push(u_id);
        }

        Ok(hg.into())
    }
}

impl Loader for WikiStdWeightedLoader {
    type Output = WeightedHypergraph;

    const VARIANT: &'static str = "w";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        // Reuse Unweighted parse and increment weights
        let contents = read_to_string(dataset_location)?;
        let mut votes: HashMap<String, Vec<String>> = HashMap::new();
        let mut hg = Hypergraph::new();

        for line in contents.lines() {
            let l = line.trim();
            if l.is_empty() {
                for (_k, v) in votes.drain() {
                    let mut uids: Vec<NodeId> = v
                        .into_iter()
                        .filter_map(|s| s.parse::<NodeId>().ok())
                        .collect();
                    let order = uids.len();
                    seq!(N in 2..11 {
                        match order {
                            N => {
                                let mut arr = [0 as NodeId; N];
                                arr.copy_from_slice(&uids);
                                if !hg.has_hyperedge(&arr) { hg.add_edge(Hx::new(arr, 0.0).expect("wiki: found malformed hyperedge")); }
                                hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                            },
                            _ => ()
                        }
                    });
                }
                continue;
            }
            if !l.starts_with('V') {
                continue;
            }
            let parts: Vec<&str> = l.split_whitespace().collect();
            if parts.len() < 3 {
                continue;
            }
            let vote = parts[1].to_string();
            let u_id = parts[2].to_string();
            votes.entry(vote).or_insert_with(Vec::new).push(u_id);
        }

        Ok(hg.into())
    }
}
