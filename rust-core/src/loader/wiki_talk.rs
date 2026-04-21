use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::graph::UnweightedHypergraph;
use crate::loader::common::{get_dataset_paths, parse};

const PATH: &str = "wiki/wiki-talk.txt";

// #[timed]
pub fn load_wiki_talk<P1, P2>(
    dataset_dir: &P1,
    cache_dir: Option<&P2>,
) -> Result<UnweightedHypergraph, Box<dyn Error>>
where
    P1: AsRef<Path> + ?Sized,
    P2: AsRef<Path> + ?Sized,
{
    if let Some(cache_dir) = cache_dir {
        load_wiki_talk_cached(dataset_dir, cache_dir)
    } else {
        Ok(load_wiki_talk_uncached(&dataset_dir)?)
    }
}

fn load_wiki_talk_cached<P1, P2>(
    dataset_dir: &P1,
    cache_dir: &P2,
) -> Result<UnweightedHypergraph, Box<dyn Error>>
where
    P1: AsRef<Path> + ?Sized,
    P2: AsRef<Path> + ?Sized,
{
    let (_dataset_path, cache_path) = get_dataset_paths(dataset_dir, cache_dir, PATH)?;

    if cache_path.exists() {
        UnweightedHypergraph::load_from_file(cache_path)
    } else {
        let rv = load_wiki_talk_uncached(dataset_dir)?;
        rv.save_to_file(cache_path)?;
        Ok(rv)
    }
}

fn load_wiki_talk_uncached<P1>(dataset_dir: &P1) -> Result<UnweightedHypergraph, Box<dyn Error>>
where
    P1: AsRef<Path> + ?Sized,
{
    let dataset_path = dataset_dir.as_ref().join(PATH);

    println!("Loading wiki-talk dataset from {:?}...", dataset_path);
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

    let mut rv = UnweightedHypergraph::new();
    rv.extends_h2(edges);

    Ok(rv)
}
