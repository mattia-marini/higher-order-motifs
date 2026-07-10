use std::error::Error;
use std::sync::LazyLock;

use rust_core::loader::DatasetLoader;
use rust_core::misc::cycle::{count_c4, intensity_c4};
use rust_core::misc::hyper_degeneracy_ordering;
use rust_core::motifs::algorithms::escape;
use rust_core::types::adj_list::{AdjList, Undirected, WithoutIncidence};
use rust_core::types::hyperadj_list::HyperAdjList;
use seq_macro::seq;

pub fn main() -> Result<(), Box<dyn Error>> {
    // simple_graph_count();
    // simple_graph_intensity();

    dblp()?;
    Ok(())
}

const SIMPLE_UNWEIGHTED: LazyLock<AdjList<(), Undirected, WithoutIncidence>> =
    LazyLock::new(|| {
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (2, 0), (0, 4), (4, 2)]
            .into_iter()
            .map(|(u, v)| (u, v, ()))
            .collect::<Vec<_>>();

        let (adj, _, _) = AdjList::from_edges_mapped(edges);

        adj
    });

const SIMPLE_WEIGHTED: LazyLock<AdjList<f64, Undirected, WithoutIncidence>> = LazyLock::new(|| {
    let edges = vec![
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 0, 1.0),
        (2, 0, 1.0),
        (0, 4, 1.0),
        (4, 2, 1.0),
    ];

    let (adj, _, _) = AdjList::from_edges_mapped(edges);

    adj
});

pub fn simple_graph_count() {
    let mut adj = SIMPLE_UNWEIGHTED.clone();
    println!("n: {}, m: {}", adj.n(), adj.m());
    println!("Non induced c4 count {:?}", count_c4(&mut adj));
}

pub fn simple_graph_intensity() {
    let mut adj = SIMPLE_WEIGHTED.clone();

    println!("n: {}, m: {}", adj.n(), adj.m());
    println!("Non induced mean c4 intensity {:?}", intensity_c4(&mut adj));
}

pub fn dblp() -> Result<(), Box<dyn Error>> {
    let mut hg = DatasetLoader::builder()
        .cached(true)
        .dblp()
        .unweighted()
        .load()?;

    seq!(N in 3..11 {
        hg.take_edges::<N>();
    });
    hg.remove_isolated_nodes();

    let (hyperadj, _, _) = HyperAdjList::<()>::from_hypergraph_mapped(hg.0);

    let t = std::time::Instant::now();
    // escape::unweighted_4(&hyperadj);
    let (_, _, k) = hyper_degeneracy_ordering(&hyperadj);
    println!("{k}");
    println!("dblp c3 count time: {:?}", t.elapsed());

    Ok(())
}
