use std::time::Duration;

use foldhash::fast::FixedState;
use hashbrown::{HashMap, HashSet};
use indicatif::ProgressIterator;

use crate::compressed_motif::{CompactMotif, CompactMotifConfigurator};

#[test]
fn test_order3() -> Result<(), Box<dyn std::error::Error>> {
    let rv = compute_all_fingerprints::<3>()?;
    assert_eq!(rv.clashing_count, 0);
    Ok(())
}

#[test]
fn test_order3() -> Result<(), Box<dyn std::error::Error>> {
    let rv = compute_all_fingerprints::<4>()?;
    assert_eq!(rv.clashing_count, 0);
    Ok(())
}
#[test]
fn test_order3() -> Result<(), Box<dyn std::error::Error>> {
    let rv = compute_all_fingerprints::<5>()?;
    assert_eq!(rv.clashing_count, 0);
    Ok(())
}

impl std::fmt::Display for EnumerationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Total motifs:          {}\n\
             Connected motifs:      {}\n\
             Distinct fingerprints: {}\n\
             Elapsed time:          {:?}\n\
             Clashing buckets:      {}",
            self.total_count,
            self.connected_count,
            self.distinct_fingerprints,
            self.elapsed_time,
            self.clashing_buckets_count
        )
    }
}

pub fn print_static_const<const N: usize>()
where
    CompactMotif<N>: CompactMotifConfigurator,
{
    println!("Adjacencies:");
    for cm in CompactMotif::<N>::ADJ {
        println!("{:016b}", cm.container);
    }

    println!("Node Map:");
    for cm in CompactMotif::<N>::NODE_MAP {
        println!("{:016b}", cm.nodes);
    }

    println!("Full overlaps:");
    for cm in CompactMotif::<N>::FULL_OVERLAPS {
        println!("{:016b}", cm.container);
    }
    println!("Part overlaps:");
    for cm in CompactMotif::<N>::PART_OVERLAPS {
        println!("{:016b}", cm.container);
    }
    println!("Edge filter bitmask");
    for cm in CompactMotif::<N>::EDGE_FILTER_BITMASK {
        println!("{:016b}", cm.container);
    }

    println!("Relabeling map");
    for cm in CompactMotif::<N>::RELABELING_MAP {
        println!("{:?}", cm);
    }
}

pub fn print_hyperedges<const N: usize>() {
    iter_hyperedges!(N, 1..=N, |edge, edge_size, edge_idx| {
        let mut v = Vec::new();
        for i in 0..edge_size {
            v.push(edge[i]);
        }
        println!("Edge {}: {:?}", edge_idx, v);
    });
}

pub fn compute_all_fingerprints<const N: usize>()
-> Result<EnumerationStats, Box<dyn std::error::Error>>
where
    CompactMotif<N>: CompactMotifConfigurator,
{
    let mut map = HashMap::with_hasher(FixedState::default());
    let time = std::time::Instant::now();

    let mut total_count = 0;
    let mut connected_count = 0;
    println!("Enumerating motifs and computing fingerprints...");

    for m in CompactMotif::<N>::enum_motifs(2..=N).progress() {
        if m.is_connected() {
            let fingerprint = m.fingerprint();
            if !map.contains_key(&fingerprint) {
                map.insert(m.fingerprint(), vec![]);
            }
            map.get_mut(&fingerprint).unwrap().push(m);
            connected_count += 1;
        }
        total_count += 1;
    }
    let mut elapsed_time = time.elapsed();

    println!("Aggregating results and checking for clashing buckets...");
    let mut clashing_buckets = Vec::new();
    for (fingerprint, motifs) in map.iter().progress() {
        let mut unique_motifs = HashSet::new();
        for motif in motifs {
            unique_motifs.insert(*motif);
        }

        let mut isomorphism = HashSet::new();
        motifs[0].enum_isomorphism(|iso| {
            isomorphism.insert(iso);
        });

        if isomorphism != unique_motifs {
            clashing_buckets.push((fingerprint, motifs));
        }
    }

    let rv = EnumerationStats {
        total_count,
        connected_count,
        elapsed_time,
        distinct_fingerprints: map.len(),
        clashing_buckets_count: clashing_buckets.len(),
    };

    Ok(rv)
}
