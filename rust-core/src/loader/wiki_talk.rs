use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::graph::Hypergraph;
use crate::graph::types::H2;
use crate::loader::common::{get_dataset_paths, parse};

const PATH: &str = "wiki-talk.txt";

// #[timed]
pub fn load_wiki_talk_cached<P1, P2>(
    dataset_dir: &P1,
    cache_dir: Option<&P2>,
) -> Result<Hypergraph, Box<dyn Error>>
where
    P1: AsRef<Path> + ?Sized,
    P2: AsRef<Path> + ?Sized,
{
    if let Some(cache_dir) = cache_dir {
        let (dataset_path, cache_path) = get_dataset_paths(dataset_dir, cache_dir, PATH)?;

        if cache_path.exists() {
            Hypergraph::load_from_file(cache_path)
        } else {
            let rv = load_wiki_talk(&dataset_path)?;
            rv.save_to_file(cache_path)?;
            Ok(rv)
        }
    } else {
        Ok(load_wiki_talk(&dataset_dir)?)
    }
}

pub fn load_wiki_talk<P1>(dataset_dir: &P1) -> Result<Hypergraph, Box<dyn Error>>
where
    P1: AsRef<Path> + ?Sized,
{
    let dataset_path = dataset_dir.as_ref().join(PATH);
    let file = File::open(dataset_path)?;

    let mut reader = BufReader::with_capacity(1_048_576, file); // 1MB buffer for faster I/O
    let mut file = Vec::with_capacity(100_000_000);

    let mut line_buf = String::new();
    reader.read_line(&mut line_buf)?;

    let parts: Vec<&str> = line_buf.trim().split_ascii_whitespace().collect();
    if parts.len() != 2 {
        return Err("Invalid header line".into());
    }
    let _n = parts[0].parse::<usize>()?;
    let m = parts[1].parse::<usize>()?;

    reader.read_to_end(&mut file)?;

    let mut sizes = [0, 0];
    let mut buffs: [[u8; 256]; 2] = [[0; 256], [0; 256]];
    let mut pos = 0;
    let mut is_first = true;

    let mut edges = Vec::with_capacity(m);

    let mut file_it = file.into_iter();

    for c in file_it.by_ref() {
        if c.is_ascii_digit() {
            buffs[0][0] = c;
            sizes[0] = 1;
            break;
        }
    }

    for c in file_it {
        if c.is_ascii_whitespace() {
            if is_first {
                if pos == 1 {
                    let u = parse(&buffs[0][..sizes[0]]);
                    let v = parse(&buffs[1][..sizes[1]]);
                    edges.push((u, v));

                    pos = 0;
                    sizes[0] = 0;
                    sizes[1] = 0;
                } else {
                    pos = 1;
                }

                is_first = false;
            }
        } else {
            buffs[pos][sizes[pos]] = c;
            sizes[pos] += 1;
            is_first = true;
        }
    }

    let mut rv = Hypergraph::new();
    rv.extends_h2(edges.into_iter().map(|(u, v)| H2::new(u, v)).collect());

    Ok(rv)
}
