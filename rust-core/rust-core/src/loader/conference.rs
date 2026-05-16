use std::{
    collections::HashMap,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::{
    graph::{
        AdjList, Hypergraph, UnweightedHypergraph,
        types::{NodeId, WHx},
    },
    loader::common::get_dataset_paths,
};

const PATH: &str = "conference.dat";

// pub fn load_conference<P1, P2>(
//     dataset_dir: &P1,
//     cache_dir: Option<&P2>,
// ) -> Result<UnweightedHypergraph, Box<dyn Error>>
// where
//     P1: AsRef<Path> + ?Sized,
//     P2: AsRef<Path> + ?Sized,
// {
//     if let Some(cache_dir) = cache_dir {
//         load_conference_cached(dataset_dir, cache_dir)
//     } else {
//         load_conference_uncached(&dataset_dir)
//     }
// }

pub fn load_conference<P1, T, W>(dataset_dir: &P1) -> Result<Hypergraph<T, W>, Box<dyn Error>>
where
    P1: AsRef<Path> + ?Sized,
{
    let dataset_path = dataset_dir.as_ref().join(PATH);

    let file = File::open(dataset_path)?;
    let reader = BufReader::new(file);

    let mut edges: HashMap<usize, Vec<(NodeId, NodeId)>> = HashMap::new();
    // let mut node_map = HashMap::new();

    for line in reader.lines() {
        let l = line?;
        let parts: Vec<&str> = l.split_whitespace().collect();

        if parts.len() == 3 {
            // Parse values (t, a, b)
            let t_raw: i32 = parts[0].parse().unwrap_or(0);
            let a = parts[1].parse().unwrap_or(0);
            let b = parts[2].parse().unwrap_or(0);

            let t = t_raw - 32520;

            // let new_id = node_map.len() as NodeId;
            // node_map.entry(a).or_insert_with(|| new_id);
            //
            // let new_id = node_map.len() as NodeId;
            // node_map.entry(b).or_insert_with(|| new_id);

            edges
                .entry(t as usize)
                .or_insert_with(Vec::new)
                .push((a, b));
        }
    }

    let mut hg = Hypergraph::new();

    for edge_list in edges.into_values() {
        let mut curr = 0;
        let mut dir_node_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut rev_node_map: HashMap<NodeId, NodeId> = HashMap::new();

        for (u, v) in edge_list.iter() {
            if !dir_node_map.contains_key(u) {
                dir_node_map.insert(*u, curr);
                rev_node_map.insert(curr, *u);
                curr += 1;
            }

            if !dir_node_map.contains_key(v) {
                dir_node_map.insert(*v, curr);
                rev_node_map.insert(curr, *v);
                curr += 1;
            }
        }

        let adj_list = AdjList::from_edges_unmapped(
            edge_list
                .iter()
                .map(|(u, v)| (dir_node_map[u], dir_node_map[v]))
                .collect(),
        );

        let cliques = adj_list.enum_cliques();

        hg.extend_with_edges(
            cliques
                .into_iter()
                .map(|edge| {
                    WHx::new(
                        edge.into_iter()
                            .map(|n| rev_node_map[&n])
                            .collect::<Vec<NodeId>>(),
                    )
                    .unwrap()
                })
                .collect(),
        );
    }

    // hg.remove_multiedges();
    // println!("{:?}", hg.h2);

    Ok(hg)
}

// pub fn load_conference_cached<P1, P2>(
//     dataset_dir: &P1,
//     cache_dir: &P2,
// ) -> Result<UnweightedHypergraph, Box<dyn Error>>
// where
//     P1: AsRef<Path> + ?Sized,
//     P2: AsRef<Path> + ?Sized,
// {
//     let (_dataset_path, cache_path) = get_dataset_paths(dataset_dir, cache_dir, PATH)?;
//
//     if cache_path.exists() {
//         UnweightedHypergraph::load_from_file(cache_path)
//     } else {
//         let rv = load_conference_uncached(dataset_dir)?;
//         rv.save_to_file(cache_path)?;
//         Ok(rv)
//     }
// }
