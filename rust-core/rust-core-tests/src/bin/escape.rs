use std::error::Error;
use std::sync::LazyLock;

use rust_core::loader::DatasetLoader;
use rust_core::misc::cycle::intensity_c4;
use rust_core::motifs::algorithms::escape::{self, unweighted_4, weighted_4};
use rust_core::types::adj_list::{AdjList, Undirected, WithoutIncidence};
use rust_core::types::hyperadj_list::{HyperAdjList, HyperAdjListBase};
use rust_core::types::{Hx, Hypergraph, NodeId};
use seq_macro::seq;

pub fn main() -> Result<(), Box<dyn Error>> {
    // simple_graph_count();
    // simple_graph_intensity();

    dblp()?;
    // hospital()?;
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

const SIMPLE_UNWEIGHTED_HYPERGRAPH: LazyLock<HyperAdjList<()>> = LazyLock::new(|| {
    let mut hg = Hypergraph::new();
    // K4
    hg.extend_with_edges(vec![
        Hx::new_unchecked([0, 1], ()),
        Hx::new_unchecked([0, 2], ()),
        Hx::new_unchecked([0, 3], ()),
        Hx::new_unchecked([1, 2], ()),
        Hx::new_unchecked([1, 3], ()),
        Hx::new_unchecked([2, 3], ()),
    ]);
    //
    // // Diamond
    // hg.extend_with_edges(vec![
    //     Hx::new_unchecked([0, 1], ()),
    //     Hx::new_unchecked([0, 2], ()),
    //     Hx::new_unchecked([0, 3], ()),
    //     Hx::new_unchecked([1, 2], ()),
    //     Hx::new_unchecked([1, 3], ()),
    // ]);
    //
    // // Tailed triangle
    // hg.extend_with_edges(vec![
    //     Hx::new_unchecked([0, 1], ()),
    //     Hx::new_unchecked([0, 2], ()),
    //     Hx::new_unchecked([0, 3], ()),
    //     Hx::new_unchecked([1, 2], ()),
    // ]);
    //
    // // C4
    // hg.extend_with_edges(vec![
    //     Hx::new_unchecked([0, 1], ()),
    //     Hx::new_unchecked([0, 2], ()),
    //     Hx::new_unchecked([1, 3], ()),
    //     Hx::new_unchecked([2, 3], ()),
    // ]);
    //
    // // Star 4
    // hg.extend_with_edges(vec![
    //     Hx::new_unchecked([0, 1], ()),
    //     Hx::new_unchecked([0, 2], ()),
    //     Hx::new_unchecked([0, 3], ()),
    // ]);
    //
    // // Path 4
    // hg.extend_with_edges(vec![
    //     Hx::new_unchecked([0, 1], ()),
    //     Hx::new_unchecked([0, 2], ()),
    //     Hx::new_unchecked([1, 3], ()),
    //     Hx::new_unchecked([2, 3], ()),
    // ]);

    // hg.extend_with_edges(vec![Hx::new_unchecked([0, 1, 2], ())]);
    // hg.extend_with_edges(vec![Hx::new_unchecked([0, 1, 3], ())]);

    let adj = HyperAdjList::from_hypergraph_unmapped(hg.clone());

    adj
});

pub fn simple_graph_count() {
    let adj = SIMPLE_UNWEIGHTED_HYPERGRAPH.clone();
    let rv = unweighted_4(&adj);
    for (number, (motif, stats)) in rv.iter().enumerate() {
        println!("{}\t{}", stats.count, motif.get_canonical_rep());
    }
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
    println!("n: {}, m: {}", hyperadj.n(), hyperadj.m());
    escape::unweighted_4(&hyperadj);
    println!("Finished in: {:?}", t.elapsed());

    Ok(())
}

pub fn hospital() -> Result<(), Box<dyn Error>> {
    let mut hg = DatasetLoader::builder()
        .cached(true)
        .hospital()
        .unweighted()
        .load()?;

    seq!(N in 3..11 {
        hg.take_edges::<N>();
    });
    hg.remove_isolated_nodes();

    let (hyperadj, _, _) = HyperAdjList::<()>::from_hypergraph_mapped(hg.0);

    let t = std::time::Instant::now();
    println!("n: {}, m: {}", hyperadj.n(), hyperadj.m());
    escape::unweighted_4(&hyperadj);
    println!("Finished in: {:?}", t.elapsed());

    Ok(())
}
