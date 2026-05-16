use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use hashbrown::HashMap;
use rust_core_macros::loader;

use crate::graph::{AdjList, Hypergraph, NodeId, NodeWeight};

const PATH: &str = "conference.dat";

#[loader(cache = "conference_unweighted.bin")]
pub fn load_conference<P1>(dataset_path: &P1) -> Result<Hypergraph<NodeId, ()>, Box<dyn Error>>
where
    P1: AsRef<Path> + ?Sized,
{
    // let dataset_path = dataset_path.as_ref().join(PATH);

    println!(
        "Loading conference dataset from {}...",
        dataset_path.as_ref().display()
    );
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

        // hg.extend_with_edges(
        //     cliques
        //         .into_iter()
        //         .map(|edge| {
        //             WHx::new(
        //                 edge.into_iter()
        //                     .map(|n| rev_node_map[&n])
        //                     .collect::<Vec<NodeId>>(),
        //             )
        //             .unwrap()
        //         })
        //         .collect(),
        // );
    }

    // hg.remove_multiedges();
    // println!("{:?}", hg.h2);

    Ok(hg)
}
