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

pub struct Unweighted;
pub struct Weighted;

impl Loader for Unweighted {
    const NAME: &'static str = "UW_justice";
    type Output = Hypergraph<NodeId, ()>;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
        let file = File::open(dataset_location)?;
        let reader = BufReader::new(file);

        let mut cases: HashMap<String, HashMap<i32, Vec<NodeId>>> = HashMap::new();
        let mut nodes: HashMap<String, NodeId> = HashMap::new();
        let mut idx: NodeId = 0;

        for line in reader.lines() {
            let l = line?;
            if l.trim().is_empty() { continue; }
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() <= 55 { continue; }

            let case_id = parts[0].trim().to_string();
            let justice_name = parts[54].trim().to_string();
            let vote_raw = parts[55].trim();
            let v = match vote_raw.parse::<i32>() { Ok(x) => x, Err(_) => continue };

            let n = if let Some(&nid) = nodes.get(&justice_name) { nid } else {
                nodes.insert(justice_name.clone(), idx);
                idx += 1;
                idx - 1
            };

            let entry = cases.entry(case_id).or_insert_with(HashMap::new);
            entry.entry(v).or_insert_with(Vec::new).push(n);
        }

        let mut hg = Hypergraph::new();

        for (_c, votes) in cases.into_iter() {
            for (_v, e) in votes.into_iter() {
                if e.len() > 1 && e.len() <= 10 {
                    seq!(N in 2..11 {
                        if e.len() == N {
                            let mut arr = [0 as NodeId; N];
                            for i in 0..N { arr[i] = e[i]; }
                            hg.add_edge(Hx::new_unchecked(arr, ()));
                        }
                    });
                }
            }
        }

        Ok(hg)
    }
}

impl Loader for Weighted {
    const NAME: &'static str = "W_justice";
    type Output = Hypergraph<NodeId, NodeWeight>;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
        let file = File::open(dataset_location)?;
        let reader = BufReader::new(file);

        let mut cases: HashMap<String, HashMap<i32, Vec<NodeId>>> = HashMap::new();
        let mut nodes: HashMap<String, NodeId> = HashMap::new();
        let mut idx: NodeId = 0;

        for line in reader.lines() {
            let l = line?;
            if l.trim().is_empty() { continue; }
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() <= 55 { continue; }

            let case_id = parts[0].trim().to_string();
            let justice_name = parts[54].trim().to_string();
            let vote_raw = parts[55].trim();
            let v = match vote_raw.parse::<i32>() { Ok(x) => x, Err(_) => continue };

            let n = if let Some(&nid) = nodes.get(&justice_name) { nid } else {
                nodes.insert(justice_name.clone(), idx);
                idx += 1;
                idx - 1
            };

            let entry = cases.entry(case_id).or_insert_with(HashMap::new);
            entry.entry(v).or_insert_with(Vec::new).push(n);
        }

        let mut hg = Hypergraph::new();

        for (_c, votes) in cases.into_iter() {
            for (_v, e) in votes.into_iter() {
                if e.len() > 1 && e.len() <= 10 {
                    seq!(N in 2..11 {
                        if e.len() == N {
                            let mut arr = [0 as NodeId; N];
                            for i in 0..N { arr[i] = e[i]; }
                            if !hg.has_hyperedge(&arr) { hg.add_edge(Hx::new_unchecked(arr, 0.0)); }
                            hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                        }
                    });
                }
            }
        }

        Ok(hg)
    }
}
