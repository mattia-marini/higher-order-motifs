use std::error::Error;
use std::fs::read_to_string;
use std::path::Path;

use hashbrown::HashMap;
use seq_macro::seq;

use crate::{
    graph::{Hx, Hypergraph, NodeId, NodeWeight},
    loader::common::Loader,
};

pub struct Unweighted;
pub struct Weighted;

impl Loader for Unweighted {
    const NAME: &'static str = "UW_wiki";
    type Output = Hypergraph<NodeId, ()>;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
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

        Ok(hg)
    }
}

impl Loader for Weighted {
    const NAME: &'static str = "W_wiki";
    type Output = Hypergraph<NodeId, NodeWeight>;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
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

        Ok(hg)
    }
}
