use foldhash::fast::FixedState;
use hashbrown::{HashMap, HashSet};
use seq_macro::seq;

use crate::{
    graph::{Hypergraph, NodeId},
    motifs::{base::*, types::compute_constants_u32},
};

// #[test]
fn test_canonical_representative_is_in_isomorphisms() {
    let hg: UnweightedHypergraph = vec![vec![1, 2], vec![2, 3]];
    let rep = get_canonical_representative(&hg, Some(3));
    let isos = enum_isomorphisms(&hg, Some(3));
    assert!(isos.contains(&rep));
}

// #[test]
fn test_generate_motifs() {
    let motifs3 = generate_motifs(3);
    let motifs4 = generate_motifs(4);
    // let motifs5 = generate_motifs(5);
    // println!("{}", motifs5.0.len());
    assert_eq!(motifs3.0.len(), 6);
    assert_eq!(motifs4.0.len(), 171);
}

fn print_motifs(motif: &Hypergraph<NodeId, ()>) {
    let mut rv: Vec<String> = Vec::new();
    seq!(N in 2..6 {
        for edge in motif.edges::<N>() {
            // rv.push_str(&format!("{:?} ", edge.nodes));
            rv.push(format!("{:?} ", edge.nodes));
        }
    });
    println!("Motif: {}", rv.join(","));
    // rv.join(",")
}

pub fn get_canonical_reps<const N: usize>(
    labelings: HashMap<UnweightedHypergraph, UnweightedHypergraph, FixedState>,
) -> HashMap<
    Vec<Vec<NodeId>>,
    (
        Vec<Hypergraph<NodeId, ()>>,
        HashMap<CanonicalRep<N, NodeId>, Vec<Hypergraph<NodeId, ()>>>,
    ),
>
where
    CanonicalRep<N, NodeId>: From<Hypergraph<NodeId, ()>>,
{
    let mut labelings_by_canonical_rep: HashMap<Vec<Vec<NodeId>>, Vec<Hypergraph<NodeId, ()>>> =
        HashMap::new();

    // let mut canonical_reps = HashMap::new();
    for (labeling, canonical_rep) in labelings.iter() {
        let labeling = labeling
            .into_iter()
            .map(|edge| edge.into_iter().map(|n| (n - 1) as NodeId).collect())
            .collect::<Vec<Vec<NodeId>>>();
        let canonical_rep = canonical_rep
            .into_iter()
            .map(|edge| edge.into_iter().map(|n| (n - 1) as NodeId).collect())
            .collect::<Vec<Vec<NodeId>>>();

        let mut hg = Hypergraph::new();
        for edge in labeling {
            hg.add_edge_vec((edge, ()));
        }

        labelings_by_canonical_rep
            .entry(canonical_rep.clone())
            .and_modify(|v: &mut Vec<Hypergraph<NodeId, ()>>| v.push(hg.clone()))
            .or_insert(vec![hg.clone()]);
    }

    let mut rv: HashMap<
        Vec<Vec<NodeId>>,
        (
            Vec<Hypergraph<NodeId, ()>>,
            HashMap<CanonicalRep<N, NodeId>, Vec<Hypergraph<NodeId, ()>>>,
        ),
    > = HashMap::new();

    for (canonical_rep, labelings) in labelings_by_canonical_rep.into_iter() {
        let mut inner_canonical_rep = HashMap::new();
        for labeling in labelings.iter() {
            let cr = CanonicalRep::from(labeling.clone());
            inner_canonical_rep
                .entry(cr.clone())
                .and_modify(|v: &mut Vec<Hypergraph<NodeId, ()>>| v.push(labeling.clone()))
                .or_insert(vec![labeling.clone()]);
        }
        rv.insert(
            canonical_rep.clone(),
            (labelings.clone(), inner_canonical_rep.clone()),
        );
    }

    let mut clashing_count = 0;
    for (canonical_rep, (labelings, inner_canonical_reps)) in rv.iter() {
        println!("Canonical rep: {:?}", canonical_rep);
        println!("\tLabelings count: {}", labelings.len());
        if inner_canonical_reps.len() > 1 {
            println!("\tCLASHING SET FOUND: {}", inner_canonical_reps.len());
            clashing_count += 1;
            for (i, (cr, reps)) in inner_canonical_reps.iter().enumerate() {
                println!("\t\t Fingerprint: {:?}", cr.degree);
                for (j, rep) in reps.iter().enumerate() {
                    print!("\t\t\t{j}");
                    print_motifs(rep);
                }
            }
        }
        // let clushing_reps = inner_canonical_reps
        //     .iter()
        //     .filter(|(_, reps)| reps.len() > 1)
        //     .collect::<Vec<_>>();
    }
    println!("{}", rv.len());
    println!("Total clashing sets: {}", clashing_count);

    rv
}

fn print_canonical_reps<const N: usize>(
    exprected_reps: Vec<Vec<Vec<usize>>>,
    gotten_reps: &HashMap<CanonicalRep<N, NodeId>, Vec<CanonicalRep<N, NodeId>>>,
) {
    println!("ORDER {}:", N);
    print!(
        "\tLabeling count:\t expected {}, got {}\n",
        exprected_reps.len(),
        gotten_reps.len()
    );

    // let clushing_reps3 = gotten_reps
    //     .iter()
    //     .filter(|(_, reps)| reps.len() > 1)
    //     .collect::<Vec<_>>();
    // print!("\tClushing sets len:\t {}\n", clushing_reps3.len());
}

#[test]
fn test_canonical_rep() {
    let (canonical3, labeling3) = generate_motifs(3);
    let (canonical4, labeling4) = generate_motifs(4);

    get_canonical_reps::<3>(labeling3);
    get_canonical_reps::<4>(labeling4);
    // for (labelings, canonical_rep) in labeling3.iter() {
    //     println!("{:?}", labelings);
    // }

    // println!("{}", labeling3.len());
    // println!("{}", labeling4.len());
    // compute_constants_u32::<5>();

    // let max = 2 << 26;
    // let block_size = max / 100;
    // let mut percentage = 0;
    // let mut curr_block_size = 0;
    // for i in 0..max {
    //     curr_block_size += 1;
    //     if curr_block_size >= block_size {
    //         percentage += 1;
    //         println!("Progress: {}%", percentage);
    //         curr_block_size = 0;
    //     }
    //
    //     // println!("Progress: {}%", percentage);
    //     // let new_percentage = i * 100 / max;
    //     // if new_percentage > percentage {
    //     //     percentage = new_percentage;
    //     // }
    // }

    // let (canonical3, labeling3) = generate_motifs(3);
    // let (canonical4, labeling4) = generate_motifs(4);
    // let (canonical5, labeling5) = generate_motifs(5);

    // let canonical_reps3 = get_canonical_reps::<3>(labeling3);
    // let canonical_reps4 = get_canonical_reps::<4>(labeling4);
    // let canonical_reps5 = get_canonical_reps::<5>(labeling5);

    // print_canonical_reps(canonical3, &canonical_reps3);

    // let len1 = canonical3.len();
    // let len2 = canonical_reps.len();
    // print!("Labeling 3:expected {}, got {}\n", len1, len2);
    // assert_eq!(
    //     len1, len2,
    //     "Unique canonical 2: expected {}, got {}",
    //     len1, len2,
    // );
    //
    // let mut canonical_reps: HashMap<CanonicalRep<4, NodeId>, Vec<CanonicalRep<4, NodeId>>> =
    //     HashMap::new();
    // for (labeling, _) in labeling4 {
    //     let mut hg = Hypergraph::new();
    //     for edge in labeling.into_iter() {
    //         let edge: Vec<NodeId> = edge.into_iter().map(|n| (n - 1) as NodeId).collect();
    //         hg.add_edge_vec((edge, ()));
    //     }
    //     let cr = CanonicalRep::from::<4>(hg);
    //     canonical_reps
    //         .entry(cr.clone())
    //         .and_modify(|v| v.push(cr.clone()))
    //         .or_insert(vec![cr]);
    // }
    // let len1 = canonical4.len();
    // let len2 = canonical_reps.len();
    // print!("Labeling 4:expected {}, got {}\n", len1, len2);
    //
    // // let mut count = 0;
    // let mut clushing_sets = Vec::new();
    // for (_, reps) in canonical_reps.iter() {
    //     if reps.len() > 1 {
    //         // count += 1;
    //         clushing_sets.push(reps.clone());
    //     }
    // }
    // print!("Clushing sets len {}\n", clushing_sets.len());
    //
    // for motif in &clushing_sets[0] {
    //     print_motifs(&motif.rep);
    // }
    //
    // assert_eq!(
    //     len1, len2,
    //     "Unique canonical 4: expected {}, got {}",
    //     len1, len2,
    // );
}
