use std::error::Error;
use std::fs::{File, read_to_string};
use std::io::{BufRead, BufReader};
use std::path::Path;

use hashbrown::HashMap;
use seq_macro::seq;

use crate::{
    graph::{AdjList, Hx, Hypergraph, NodeId, NodeWeight},
    loader::common::Loader,
};

// Workspace: similar to conference but different time offset
pub mod workspace {
    use super::*;

    pub struct Unweighted;
    pub struct Weighted;

    impl Loader for Unweighted {
        const NAME: &'static str = "UW_workspace";
        type Output = Hypergraph<NodeId, ()>;

        fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
        where
            P: AsRef<Path> + ?Sized,
        {
            let file = File::open(dataset_location)?;
            let reader = BufReader::new(file);

            let mut edges: HashMap<usize, Vec<(NodeId, NodeId)>> = HashMap::new();

            for line in reader.lines() {
                let l = line?;
                let parts: Vec<&str> = l.split_whitespace().collect();
                if parts.len() == 3 {
                    let t_raw: i32 = parts[0].parse().unwrap_or(0);
                    let a = parts[1].parse().unwrap_or(0);
                    let b = parts[2].parse().unwrap_or(0);

                    let t = t_raw - 28820;
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

                seq!(N in 2..11 {
                    let mut bucket_~N: Vec<Hx<N, NodeId, ()>> = Vec::new();
                });

                for clique in cliques.into_iter() {
                    seq!(N in 2..11 {
                        match clique.len() {
                            #(N => bucket_~N.push(Hx::new(clique.try_into().expect("Tuple length problem"), ()).expect("Duplicate node in clique")),)*
                            _ => (),
                        }
                    })
                }

                seq!(N in 2..11 {
                    hg.extend_with_edges(bucket_~N);
                });
            }

            Ok(hg)
        }
    }

    impl Loader for Weighted {
        const NAME: &'static str = "W_workspace";
        type Output = Hypergraph<NodeId, NodeWeight>;

        fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
        where
            P: AsRef<Path> + ?Sized,
        {
            let file = File::open(dataset_location)?;
            let reader = BufReader::new(file);

            let mut edges: HashMap<usize, Vec<(NodeId, NodeId)>> = HashMap::new();

            for line in reader.lines() {
                let l = line?;
                let parts: Vec<&str> = l.split_whitespace().collect();
                if parts.len() == 3 {
                    let t_raw: i32 = parts[0].parse().unwrap_or(0);
                    let a = parts[1].parse().unwrap_or(0);
                    let b = parts[2].parse().unwrap_or(0);

                    let t = t_raw - 28820;
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

                seq!(N in 2..11 {
                    let mut bucket_~N: Vec<Hx<N, NodeId, ()>> = Vec::new();
                });

                for clique in cliques.into_iter() {
                    seq!(N in 2..11 {
                        match clique.len() {
                            #(N => bucket_~N.push(Hx::new(clique.try_into().expect("Tuple length problem"), ()).expect("Duplicate node in clique")),)*
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

            Ok(hg)
        }
    }
}

// DBLP / History / Geology share the same CSV (paper, author, year) layout
macro_rules! paper_author_loader {
    ($mod_name:ident, $uw_name:expr, $w_name:expr) => {
        pub mod $mod_name {
            use super::*;

            pub struct Unweighted;
            pub struct Weighted;

            impl Loader for Unweighted {
                const NAME: &'static str = $uw_name;
                type Output = Hypergraph<NodeId, ()>;

                fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
                where
                    P: AsRef<Path> + ?Sized,
                {
                    let file = File::open(dataset_location)?;
                    let reader = BufReader::new(file);

                    let mut graph: HashMap<String, Vec<NodeId>> = HashMap::new();

                    for line in reader.lines() {
                        let l = line?;
                        if l.trim().is_empty() { continue; }
                        // naive CSV split, assume no commas in fields of interest
                        let parts: Vec<&str> = l.split(',').collect();
                        if parts.len() >= 2 {
                            let paper = parts[0].to_string();
                            if let Ok(author) = parts[1].trim().parse::<NodeId>() {
                                graph.entry(paper).or_insert_with(Vec::new).push(author);
                            }
                        }
                    }

                    let mut hg = Hypergraph::new();

                    for (_paper, authors) in graph.into_iter() {
                        let mut a = authors;
                        if a.len() > 1 && a.len() <= 10 {
                            seq!(N in 2..11 {
                                if a.len() == N {
                                    let mut arr = [0 as NodeId; N];
                                    for i in 0..N { arr[i] = a[i]; }
                                    hg.add_edge(Hx::new_unchecked(arr, ()));
                                }
                            });
                        }
                    }

                    Ok(hg)
                }
            }

            impl Loader for Weighted {
                const NAME: &'static str = $w_name;
                type Output = Hypergraph<NodeId, NodeWeight>;

                fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
                where
                    P: AsRef<Path> + ?Sized,
                {
                    let file = File::open(dataset_location)?;
                    let reader = BufReader::new(file);

                    let mut graph: HashMap<String, Vec<NodeId>> = HashMap::new();

                    for line in reader.lines() {
                        let l = line?;
                        if l.trim().is_empty() { continue; }
                        let parts: Vec<&str> = l.split(',').collect();
                        if parts.len() >= 2 {
                            let paper = parts[0].to_string();
                            if let Ok(author) = parts[1].trim().parse::<NodeId>() {
                                graph.entry(paper).or_insert_with(Vec::new).push(author);
                            }
                        }
                    }

                    let mut hg = Hypergraph::new();

                    for (_paper, authors) in graph.into_iter() {
                        let mut a = authors;
                        if a.len() > 1 && a.len() <= 10 {
                            seq!(N in 2..11 {
                                if a.len() == N {
                                    let mut arr = [0 as NodeId; N];
                                    for i in 0..N { arr[i] = a[i]; }
                                    if !hg.has_hyperedge(&arr) {
                                        hg.add_edge(Hx::new_unchecked(arr, 0.0));
                                    }
                                    hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                                }
                            });
                        }
                    }

                    Ok(hg)
                }
            }
        }
    };
}

paper_author_loader!(dblp, "UW_DBLP", "W_DBLP");
paper_author_loader!(history, "UW_history", "W_history");
paper_author_loader!(geology, "UW_geology", "W_geology");

// Justice loader (ignores ideology extra map)
pub mod justice {
    use super::*;
    use std::collections::HashMap as StdMap;

    pub struct Unweighted;
    pub struct Weighted;

    impl Loader for Unweighted {
        const NAME: &'static str = "UW_justice";
        type Output = Hypergraph<NodeId, ()>;

        fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
        where
            P: AsRef<Path> + ?Sized,
        {
            // We attempt a best-effort CSV parse. The original Python used specific columns.
            let file = File::open(dataset_location)?;
            let reader = BufReader::new(file);

            let mut cases: HashMap<String, StdMap<i32, Vec<NodeId>>> = HashMap::new();
            let mut nodes: StdMap<String, NodeId> = StdMap::new();
            let mut idx: NodeId = 0;

            for line in reader.lines() {
                let l = line?;
                if l.trim().is_empty() {
                    continue;
                }
                let parts: Vec<&str> = l.split(',').collect();
                if parts.len() <= 55 {
                    continue;
                } // expect many columns

                let case_id = parts[0].trim().to_string();
                let justice_name = parts[54].trim().to_string();
                let vote_raw = parts[55].trim();
                let v = match vote_raw.parse::<i32>() {
                    Ok(x) => x,
                    Err(_) => continue,
                };

                let n = if let Some(&nid) = nodes.get(&justice_name) {
                    nid
                } else {
                    nodes.insert(justice_name.clone(), idx);
                    idx += 1;
                    idx - 1
                };

                let entry = cases.entry(case_id).or_insert_with(StdMap::new);
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
            // Reuse the Unweighted parse but accumulate weights
            let file = File::open(dataset_location)?;
            let reader = BufReader::new(file);

            let mut cases: HashMap<String, StdMap<i32, Vec<NodeId>>> = HashMap::new();
            let mut nodes: StdMap<String, NodeId> = StdMap::new();
            let mut idx: NodeId = 0;

            for line in reader.lines() {
                let l = line?;
                if l.trim().is_empty() {
                    continue;
                }
                let parts: Vec<&str> = l.split(',').collect();
                if parts.len() <= 55 {
                    continue;
                }

                let case_id = parts[0].trim().to_string();
                let justice_name = parts[54].trim().to_string();
                let vote_raw = parts[55].trim();
                let v = match vote_raw.parse::<i32>() {
                    Ok(x) => x,
                    Err(_) => continue,
                };

                let n = if let Some(&nid) = nodes.get(&justice_name) {
                    nid
                } else {
                    nodes.insert(justice_name.clone(), idx);
                    idx += 1;
                    idx - 1
                };

                let entry = cases.entry(case_id).or_insert_with(StdMap::new);
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
}

// Babbuini: gzipping reading not supported here; assume plain text input
pub mod babbuini {
    use super::*;

    pub struct Unweighted;
    pub struct Weighted;

    impl Loader for Unweighted {
        const NAME: &'static str = "UW_babbuini";
        type Output = Hypergraph<NodeId, ()>;

        fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
        where
            P: AsRef<Path> + ?Sized,
        {
            let contents = read_to_string(dataset_location)?;
            let lines: Vec<&str> = contents.lines().collect();

            let mut graph: HashMap<i32, Vec<(NodeId, NodeId)>> = HashMap::new();
            let mut names: HashMap<String, NodeId> = HashMap::new();
            let mut idx: NodeId = 0;
            let mut cont = 0;

            for l in lines.iter() {
                if cont == 0 {
                    cont = 1;
                    continue;
                }
                let parts: Vec<&str> = l.split_whitespace().collect();
                if parts.len() < 3 {
                    continue;
                }
                let t: i32 = parts[0].parse().unwrap_or(0);
                let a_s = parts[1].to_string();
                let b_s = parts[2].to_string();

                let a = if let Some(&v) = names.get(&a_s) {
                    v
                } else {
                    names.insert(a_s.clone(), idx);
                    idx += 1;
                    idx - 1
                };
                let b = if let Some(&v) = names.get(&b_s) {
                    v
                } else {
                    names.insert(b_s.clone(), idx);
                    idx += 1;
                    idx - 1
                };

                graph.entry(t).or_insert_with(Vec::new).push((a, b));
            }

            let mut hg = Hypergraph::new();

            for (_t, edge_list) in graph.into_iter() {
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
                            #(N => bucket_~N.push(Hx::new(clique.try_into().expect("Tuple length problem"), ()).expect("Duplicate node in clique")),)*
                            _ => (),
                        }
                    })
                }

                seq!(N in 2..11 { hg.extend_with_edges(bucket_~N); });
            }

            Ok(hg)
        }
    }

    impl Loader for Weighted {
        const NAME: &'static str = "W_babbuini";
        type Output = Hypergraph<NodeId, NodeWeight>;

        fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
        where
            P: AsRef<Path> + ?Sized,
        {
            // Reuse Unweighted parse and add weights
            let contents = read_to_string(dataset_location)?;
            let lines: Vec<&str> = contents.lines().collect();

            let mut graph: HashMap<i32, Vec<(NodeId, NodeId)>> = HashMap::new();
            let mut names: HashMap<String, NodeId> = HashMap::new();
            let mut idx: NodeId = 0;
            let mut cont = 0;

            for l in lines.iter() {
                if cont == 0 {
                    cont = 1;
                    continue;
                }
                let parts: Vec<&str> = l.split_whitespace().collect();
                if parts.len() < 3 {
                    continue;
                }
                let t: i32 = parts[0].parse().unwrap_or(0);
                let a_s = parts[1].to_string();
                let b_s = parts[2].to_string();

                let a = if let Some(&v) = names.get(&a_s) {
                    v
                } else {
                    names.insert(a_s.clone(), idx);
                    idx += 1;
                    idx - 1
                };
                let b = if let Some(&v) = names.get(&b_s) {
                    v
                } else {
                    names.insert(b_s.clone(), idx);
                    idx += 1;
                    idx - 1
                };

                graph.entry(t).or_insert_with(Vec::new).push((a, b));
            }

            let mut hg = Hypergraph::new();

            for (_t, edge_list) in graph.into_iter() {
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
                            #(N => bucket_~N.push(Hx::new(clique.try_into().expect("Tuple length problem"), ()).expect("Duplicate node in clique")),)*
                            _ => (),
                        }
                    })
                }

                seq!(N in 2..11 {
                    for edge in bucket_~N.into_iter() {
                        if !hg.has_hyperedge(&edge.nodes) { hg.add_edge(Hx::new_unchecked(edge.nodes, 0.0)); }
                        hg.modify_hx_weigth_with(&edge.nodes, |w| w + 1.0);
                    }
                });
            }

            Ok(hg)
        }
    }
}

// Wiki loader (vote blocks)
pub mod wiki {
    use super::*;

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
                        let mut uids: Vec<NodeId> = v
                            .into_iter()
                            .filter_map(|s| s.parse::<NodeId>().ok())
                            .collect();
                        if uids.len() > 1 && uids.len() <= 10 {
                            seq!(N in 2..11 {
                                if uids.len() == N {
                                    let mut arr = [0 as NodeId; N];
                                    for i in 0..N { arr[i] = uids[i]; }
                                    hg.add_edge(Hx::new_unchecked(arr, ()));
                                }
                            });
                        }
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
                        if uids.len() > 1 && uids.len() <= 10 {
                            seq!(N in 2..11 {
                                if uids.len() == N {
                                    let mut arr = [0 as NodeId; N];
                                    for i in 0..N { arr[i] = uids[i]; }
                                    if !hg.has_hyperedge(&arr) { hg.add_edge(Hx::new_unchecked(arr, 0.0)); }
                                    hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                                }
                            });
                        }
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
}

// NDC substances / classes and EU / Enron share simplex files
macro_rules! simplices_loader {
    ($mod_name:ident, $uw_name:expr, $w_name:expr) => {
        pub mod $mod_name {
            use super::*;

            pub struct Unweighted;
            pub struct Weighted;

            fn read_ints_from_file<P: AsRef<Path>>(path: &P) -> Result<Vec<NodeId>, Box<dyn Error>> {
                let s = read_to_string(path)?;
                let mut v = Vec::new();
                for line in s.lines() {
                    let t = line.trim(); if t.is_empty() { continue; }
                    if let Ok(x) = t.parse::<NodeId>() { v.push(x); }
                }
                Ok(v)
            }

            impl Loader for Unweighted {
                const NAME: &'static str = $uw_name;
                type Output = Hypergraph<NodeId, ()>;

                fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
                where
                    P: AsRef<Path> + ?Sized,
                {
                    let base = dataset_location.as_ref();
                    let nverts_path = if base.is_dir() { base.join(format!("{}-nverts.txt", base.file_name().and_then(|n| n.to_str()).unwrap_or(""))) } else { base.with_extension("-nverts.txt") };
                    let simplices_path = if base.is_dir() { base.join(format!("{}-simplices.txt", base.file_name().and_then(|n| n.to_str()).unwrap_or(""))) } else { base.with_extension("-simplices.txt") };

                    let v = read_ints_from_file(&nverts_path)?;
                    let mut s = read_ints_from_file(&simplices_path)?;

                    let mut hg = Hypergraph::new();

                    for mut i in v.into_iter() {
                        let mut e: Vec<NodeId> = Vec::new();
                        for _ in 0..i {
                            if s.is_empty() { break; }
                            e.push(s.remove(0));
                        }
                        if e.len() > 1 && e.len() <= 10 {
                            seq!(N in 2..11 {
                                if e.len() == N {
                                    let mut arr = [0 as NodeId; N];
                                    for j in 0..N { arr[j] = e[j]; }
                                    hg.add_edge(Hx::new_unchecked(arr, ()));
                                }
                            });
                        }
                    }

                    Ok(hg)
                }
            }

            impl Loader for Weighted {
                const NAME: &'static str = $w_name;
                type Output = Hypergraph<NodeId, NodeWeight>;

                fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
                where
                    P: AsRef<Path> + ?Sized,
                {
                    let base = dataset_location.as_ref();
                    let nverts_path = if base.is_dir() { base.join(format!("{}-nverts.txt", base.file_name().and_then(|n| n.to_str()).unwrap_or(""))) } else { base.with_extension("-nverts.txt") };
                    let simplices_path = if base.is_dir() { base.join(format!("{}-simplices.txt", base.file_name().and_then(|n| n.to_str()).unwrap_or(""))) } else { base.with_extension("-simplices.txt") };

                    let v = read_ints_from_file(&nverts_path)?;
                    let mut s = read_ints_from_file(&simplices_path)?;

                    let mut hg = Hypergraph::new();

                    for mut i in v.into_iter() {
                        let mut e: Vec<NodeId> = Vec::new();
                        for _ in 0..i {
                            if s.is_empty() { break; }
                            e.push(s.remove(0));
                        }
                        if e.len() > 1 && e.len() <= 10 {
                            seq!(N in 2..11 {
                                if e.len() == N {
                                    let mut arr = [0 as NodeId; N];
                                    for j in 0..N { arr[j] = e[j]; }
                                    if !hg.has_hyperedge(&arr) { hg.add_edge(Hx::new_unchecked(arr, 0.0)); }
                                    hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                                }
                            });
                        }
                    }

                    Ok(hg)
                }
            }
        }
    };
}

simplices_loader!(ndc_substances, "UW_NDC_substances", "W_NDC_substances");
simplices_loader!(ndc_classes, "UW_NDC_classes", "W_NDC_classes");
simplices_loader!(eu, "UW_eu", "W_eu");
simplices_loader!(enron, "UW_enron", "W_enron");

// Note: wiki_talk already has a dedicated implementation in its own file; we do not
// reimplement it here.
