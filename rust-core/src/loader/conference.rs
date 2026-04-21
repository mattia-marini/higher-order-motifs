use std::{
    collections::HashMap,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::{
    graph::{AdjList, UnweightedHypergraph, types::NodeId},
    misc::clique_3_cloj,
};

pub fn load_conference<P1, P2>(
    dataset_dir: &P1,
    cache_dir: Option<&P2>,
) -> Result<UnweightedHypergraph, Box<dyn Error>>
where
    P1: AsRef<Path> + ?Sized,
    P2: AsRef<Path> + ?Sized,
{
    if let Some(cache_dir) = cache_dir {
        load_conference_cached(dataset_dir, cache_dir)
    } else {
        Ok(load_conference_uncached(&dataset_dir)?)
    }
}

pub fn load_conference_uncached<P1>(dataset_dir: &P1) -> Result<UnweightedHypergraph, Box<dyn Error>>
where
    P1: AsRef<Path> + ?Sized,
{
    let file = File::open(dataset_dir)?;
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

    let mut graph = UnweightedHypergraph::new();

    for (t, edge_list) in edges {
        let mut adj_list = AdjList::from_edges(&edge_list);
        adj_list.make_undirected();

        clique_3_cloj(adj_list, true, cloj);
    }

    graph.extends_h2(edges);
    Ok(graph)
}

pub fn load_conference_cached<P1, P2>(
    dataset_dir: &P1,
    cache_dir: &P2,
) -> Result<UnweightedHypergraph, Box<dyn Error>>
where
    P1: AsRef<Path> + ?Sized,
    P2: AsRef<Path> + ?Sized,
{
}
