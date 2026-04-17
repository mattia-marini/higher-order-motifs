use super::bin_writer::save_vec;
use std::cmp::{self, max};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use timed::timed;

use crate::graph::FlatAdjList;
use crate::graph::bin_writer::load_vec;

use super::adj_list::AdjList;

#[inline(always)]
pub fn parse(chars: &[u8]) -> u32 {
    let mut rv = 0;
    let mut base = 1;
    for c in chars.iter().rev() {
        rv += (c - b'0') as u32 * base;
        base *= 10;
    }
    rv
}

#[timed]
pub fn load_wiki_talk() -> Result<FlatAdjList<u32, u32>, Box<dyn Error>> {
    let path =
        "/Users/mattia/Desktop/workspaces/python/higher-order-motifs/dataset/wiki/wiki-talk.txt";

    let cache =
        "/Users/mattia/Desktop/workspaces/python/higher-order-motifs/dataset/wiki/wiki-talk.bin";

    if Path::new(cache).exists() {
        let rv = FlatAdjList::load_from_file(cache)?;
        // println!("{:?}", &edges[0..100]);
        Ok(rv)
    } else {
        let file = File::open(path).map_err(|e| e.to_string())?;

        let mut reader = BufReader::with_capacity(1_048_576, file); // 1MB buffer for faster I/O
        let mut file = Vec::with_capacity(100_000_000);

        let mut line_buf = String::new();
        reader.read_line(&mut line_buf)?;

        let parts: Vec<&str> = line_buf.trim().split_ascii_whitespace().collect();
        let (n, m) = if parts.len() == 2 {
            if let (Ok(u), Ok(v)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                (u, v)
            } else {
                return Err("Invalid header line".into());
            }
        } else {
            return Err("Invalid header line".into());
        };
        println!("Header: n {} m {}", n, m);

        reader.read_to_end(&mut file);

        // let v: Vec<char> = file[0..100].iter().map(|c| *c as char).collect();
        // println!("{:?}", v);

        let mut sizes = [0, 0];
        let mut buffs: [[u8; 256]; 2] = [[0; 256], [0; 256]];
        let mut pos = 0;
        let mut is_first = true;

        let mut edges = Vec::with_capacity(100_000_000);

        let mut file_it = file.into_iter();

        while let Some(c) = file_it.next() {
            if c.is_ascii_digit() {
                buffs[0][0] = c;
                sizes[0] = 1;
                break;
            }
        }

        // let mut adj_mat = AdjMat::with_nodes(n);
        for c in file_it {
            if (c == b'\t' || c == b'\r' || c == b'\n') {
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
        println!("Finished parsing edges. Got {} edges", edges.len());

        let rv = FlatAdjList::from_edges(&edges, false);
        rv.save_to_file(cache)?;

        Ok(rv)
    }

    // Ok(FlatAdjList::new())
}
