use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use hashbrown::HashMap;
use seq_macro::seq;

use crate::{
    graph::{
        AdjList, Hx, Hypergraph, NodeId, NodeWeight, UnweightedHypergraph, WeightedHypergraph,
    },
    loader::{common::Loader, error::LoaderError},
    misc::find_cliques,
};

use super::{ConferenceStdUnweightedLoader, ConferenceStdWeightedLoader};

impl Loader for ConferenceStdUnweightedLoader {
    type Output = UnweightedHypergraph;

    const VARIANT: &'static str = "uw";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        let file = File::open(dataset_location)?;
        let reader = BufReader::new(file);

        let mut edges: HashMap<usize, Vec<(NodeId, NodeId)>> = HashMap::new();

        for line in reader.lines() {
            let l = line?;
            let parts: Vec<&str> = l.split_whitespace().collect();

            if parts.len() == 3 {
                // Parse values (t, a, b)
                let t_raw: i32 = parts[0].parse().unwrap_or(0);
                let a = parts[1].parse().unwrap_or(0);
                let b = parts[2].parse().unwrap_or(0);

                let t = t_raw - 32520;

                edges
                    .entry(t as usize)
                    .or_insert_with(Vec::new)
                    .push((a, b));
            }
        }

        let mut hg = Hypergraph::new();

        for (t, edge_list) in edges.into_iter() {
            let len = edge_list.len();
            let (mut adj_list, original_index, compressed_index) = AdjList::from_edges_mapped(
                edge_list.into_iter().map(|(u, v)| (u, v, ())).collect(),
                // .iter()
                // .map(|(u, v)| (dir_node_map[u], dir_node_map[v]))
                // .collect(),
            );

            adj_list.make_undirected();

            let mut cliques = find_cliques(&adj_list);
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

            seq!(N in 2..11 {
                let mut bucket_~N: Vec<Hx<N, NodeId, ()>> = Vec::new();
            });

            for clique in cliques.into_iter() {
                seq!(N in 2..11 {
                    match clique.len() {
                        #(N => bucket_~N.push(Hx::new(clique.try_into().expect("Tuple length problem. Should not happen"), ()).expect("Clique found with duplicate node")),)*
                        _ => (),
                    }
                })
            }
            seq!(N in 2..11 {
                hg.extend_with_edges(bucket_~N);
            });
        }

        Ok(hg.into())
    }
}

impl Loader for ConferenceStdWeightedLoader {
    type Output = WeightedHypergraph;

    const VARIANT: &'static str = "w";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        let file = File::open(dataset_location)?;
        let reader = BufReader::new(file);

        let mut edges: HashMap<usize, Vec<(NodeId, NodeId)>> = HashMap::new();

        for line in reader.lines() {
            let l = line?;
            let parts: Vec<&str> = l.split_whitespace().collect();

            if parts.len() == 3 {
                // Parse values (t, a, b)
                let t_raw: i32 = parts[0].parse().unwrap_or(0);
                let a = parts[1].parse().unwrap_or(0);
                let b = parts[2].parse().unwrap_or(0);

                let t = t_raw - 32520;

                edges
                    .entry(t as usize)
                    .or_insert_with(Vec::new)
                    .push((a, b));
            }
        }

        let mut hg = Hypergraph::new();

        for (t, edge_list) in edges.into_iter() {
            let len = edge_list.len();
            let (mut adj_list, original_index, compressed_index) = AdjList::from_edges_mapped(
                edge_list.into_iter().map(|(u, v)| (u, v, ())).collect(),
            );

            adj_list.make_undirected();

            let mut cliques = find_cliques(&adj_list);
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

            seq!(N in 2..11 {
                let mut bucket_~N: Vec<Hx<N, NodeId, ()>> = Vec::new();
            });

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
