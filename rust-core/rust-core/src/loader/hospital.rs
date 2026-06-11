use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use hashbrown::HashMap;
use seq_macro::seq;

use crate::{
    graph::{AdjList, Hx, Hypergraph, NodeId, NodeWeight, WeightedHypergraph},
    loader::common::DatasetInfo,
};

use super::Loader;
use super::{HospitalStdUnweightedLoader, HospitalStdWeightedLoader};

impl Loader for HospitalStdUnweightedLoader {
    type Output = crate::graph::UnweightedHypergraph;

    fn from_file(&self) -> Result<Self::Output, Box<dyn Error>> {
        let dataset_location = self.dataset_location.clone();
        let file = File::open(dataset_location)?;
        let reader = BufReader::new(file);

        let mut edges: HashMap<usize, Vec<(NodeId, NodeId)>> = HashMap::new();

        for line in reader.lines() {
            let l = line?;
            let parts: Vec<&str> = l.split_whitespace().collect();
            if parts.len() >= 3 {
                let t_raw: i32 = parts[0].parse().unwrap_or(0);
                let a = parts[1].parse().unwrap_or(0);
                let b = parts[2].parse().unwrap_or(0);
                let t = t_raw - 140;
                edges
                    .entry(t as usize)
                    .or_insert_with(Vec::new)
                    .push((a, b));
            }
        }

        let mut hg = Hypergraph::new();

        for (_t, edge_list) in edges.into_iter() {
            let (mut adj_list, original_index, _compressed_index) =
                AdjList::from_edges_mapped(edge_list);
            adj_list.make_undirected();
            let mut cliques = adj_list.find_cliques();
            cliques = cliques
                .into_iter()
                .filter(|c| c.len() >= 2)
                .map(|clique| {
                    clique
                        .into_iter()
                        .map(|node| original_index[node as usize])
                        .collect()
                })
                .collect();

            seq!(N in 2..11 { let mut bucket_~N: Vec<Hx<N, NodeId, ()>> = Vec::new(); });

            for clique in cliques.into_iter() {
                seq!(N in 2..11 {
                    match clique.len() {
                        #(N => bucket_~N.push(Hx::new(clique.try_into().expect("Tuple length problem. Should not happen"), ()).expect("Clique found with duplicate node")),)*
                        _ => (),
                    }
                })
            }

            seq!(N in 2..11 { hg.extend_with_edges(bucket_~N); });
        }

        Ok(hg.into())
    }
}

impl Loader for HospitalStdWeightedLoader {
    type Output = WeightedHypergraph;

    fn from_file(&self) -> Result<Self::Output, Box<dyn Error>> {
        let dataset_location = self.dataset_location.clone();
        let file = File::open(self.dataset_location())?;
        let reader = BufReader::new(file);

        let mut edges: HashMap<usize, Vec<(NodeId, NodeId)>> = HashMap::new();

        for line in reader.lines() {
            let l = line?;
            let parts: Vec<&str> = l.split_whitespace().collect();
            if parts.len() >= 3 {
                let t_raw: i32 = parts[0].parse().unwrap_or(0);
                let a = parts[1].parse().unwrap_or(0);
                let b = parts[2].parse().unwrap_or(0);
                let t = t_raw - 140;
                edges
                    .entry(t as usize)
                    .or_insert_with(Vec::new)
                    .push((a, b));
            }
        }

        let mut hg = Hypergraph::new();

        for (_t, edge_list) in edges.into_iter() {
            let (mut adj_list, original_index, _compressed_index) =
                AdjList::from_edges_mapped(edge_list);
            adj_list.make_undirected();
            let mut cliques = adj_list.find_cliques();
            cliques = cliques
                .into_iter()
                .filter(|c| c.len() >= 2)
                .map(|clique| {
                    clique
                        .into_iter()
                        .map(|node| original_index[node as usize])
                        .collect()
                })
                .collect();

            seq!(N in 2..11 { let mut bucket_~N: Vec<Hx<N, NodeId, ()>> = Vec::new(); });

            for clique in cliques.into_iter() {
                seq!(N in 2..11 {
                    match clique.len() {
                        #(N => bucket_~N.push(Hx::new(clique.try_into().expect("Tuple length problem. Should not happen"), ()).expect("Clique found with duplicate node")),)*
                        _ => (),
                    }
                })
            }

            seq!(N in 2..11 {
                for edge in bucket_~N.into_iter() {
                    if !hg.has_hyperedge(&edge.nodes) {
                        hg.add_edge(Hx::new_unchecked(edge.nodes, 0.0));
                    }
                    hg.modify_hx_weigth_with(&edge.nodes, |w| w + 1.0);
                }
            });
        }

        Ok(hg.into())
    }
}
