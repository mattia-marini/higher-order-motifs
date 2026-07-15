#![allow(dead_code)]
use std::time::Duration;
use std::{error::Error, time::Instant};

use rust_core::misc::{degeneracy_ordering, hyper_degeneracy_ordering};
use rust_core::motifs::algorithms::escape;
use rust_core::types::adj_list::{AdjList, Undirected, WithoutIncidence};
use rust_core::{loader::DatasetLoader, types::hyperadj_list::HyperAdjList};

/// Computes the $k$-uniform density of a hypergraph.
///
/// The density is defined as: $\rho = |E| / \binom{n}{k}$
///
/// # Arguments
/// * `num_vertices` ($n$) - The total number of vertices in the hypergraph.
/// * `num_edges` ($|E|$) - The number of active hyperedges.
/// * `k` - The uniformity of the hypergraph (each hyperedge has size `k`).
///
/// # Returns
/// * `Some(f64)` representing the density in $[0.0, 1.0]$.
/// * `None` if $k > n$, or if $n$ is 0, or if the calculation overflows.
pub fn k_uniform_density(num_vertices: usize, num_edges: usize, k: usize) -> Option<f64> {
    if k == 0 || k > num_vertices {
        return None;
    }

    // Calculate the binomial coefficient \binom{n}{k} safely
    let max_possible_edges = binomial_coefficient(num_vertices, k)?;

    if num_edges > max_possible_edges {
        // Technically impossible for a simple hypergraph, but guards against invalid input
        return None;
    }

    Some(num_edges as f64 / max_possible_edges as f64)
}

/// Helper function to compute \binom{n}{k} using a safe iterative approach
/// to prevent overflow where possible.
fn binomial_coefficient(n: usize, mut k: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    // Since \binom{n}{k} == \binom{n}{n-k}, we can optimize the loop
    if k > n - k {
        k = n - k;
    }

    let mut res = 1;
    for i in 1..=k {
        // Check for multiplication overflow before multiplying
        res *= n - k + i;
        res /= i;
    }
    Some(res)
}

#[derive(Debug, Clone)]
struct BenchmarkResult {
    dataset_name: String,
    description: String,

    weighted: OrderResult,
    unweighted: OrderResult,
}

#[derive(Debug, Clone)]
struct OrderResult {
    order3: ExecutionInfos,
    order4: ExecutionInfos,
}

#[derive(Debug, Clone)]
struct ExecutionInfos {
    time: Duration,
    graph_infos: GraphInfos,
}

impl Default for ExecutionInfos {
    fn default() -> Self {
        Self {
            time: Duration::new(0, 0),
            graph_infos: GraphInfos::default(),
        }
    }
}

impl Default for OrderResult {
    fn default() -> Self {
        Self {
            order3: ExecutionInfos::default(),
            order4: ExecutionInfos::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct GraphInfos {
    n: usize,
    e2: usize,
    e3: usize,
    e4: usize,
    density2: f64,
    density3: f64,
    density4: f64,
    max_degree: usize,
    degeneracy: usize,
    hyper_degeneracy: usize,
}

impl Default for GraphInfos {
    fn default() -> Self {
        Self {
            n: 0,
            e2: 0,
            e3: 0,
            e4: 0,
            density2: 0.0,
            density3: 0.0,
            density4: 0.0,
            max_degree: 0,
            degeneracy: 0,
            hyper_degeneracy: 0,
        }
    }
}

macro_rules! test_dataset {
    ($name: ident, $rv: ident) => {
        println!("Running benchmark for dataset: {}", stringify!($name));

        // --- 1. UNWEIGHTED PIPELINE ---
        let mut hg = DatasetLoader::builder()
            .cached(true)
            .$name()
            .unweighted()
            .load()?;

        hg.retain_orders(vec![2, 3, 4]);

        let n = hg.n();
        let count2 = hg.0.edges::<2>().len();
        let count3 = hg.0.edges::<3>().len();
        let count4 = hg.0.edges::<4>().len();

        let (hyper_adj, _, _) = HyperAdjList::<()>::from_hypergraph_mapped(hg.0.clone());
        let mut max_degree = 0;
        for neighbors in hyper_adj.iter_all_incident_edges() {
            max_degree = max_degree.max(neighbors.len());
        }
        let (_, _, hyper_degeneracy) = hyper_degeneracy_ordering(&hyper_adj);

        let edges =
            hg.0.edges::<2>()
                .iter()
                .map(|e| (e.nodes[0], e.nodes[1], ()))
                .collect::<Vec<_>>();
        let (adj, _, _) = AdjList::<(), Undirected, WithoutIncidence>::from_edges_mapped(edges);
        let (_, _, degeneracy) = degeneracy_ordering(&adj);

        let unweighted_graph_infos = GraphInfos {
            n,
            e2: count2,
            e3: count3,
            e4: count4,
            density2: k_uniform_density(n, count2, 2).unwrap_or(0.0),
            density3: k_uniform_density(n, count3, 3).unwrap_or(0.0),
            density4: k_uniform_density(n, count4, 4).unwrap_or(0.0),
            max_degree,
            degeneracy,
            hyper_degeneracy,
        };

        // Benchmark unweighted 3
        let time = Instant::now();
        escape::unweighted_3(&hg);
        let unweighted_order3 = ExecutionInfos {
            time: time.elapsed(),
            graph_infos: unweighted_graph_infos.clone(),
        };

        // Benchmark unweighted 4
        let (adj_order4, _, _) = HyperAdjList::<()>::from_hypergraph_mapped(hg.0);
        let time = Instant::now();
        escape::unweighted_4(&adj_order4);
        let order4 = ExecutionInfos {
            time: time.elapsed(),
            graph_infos: unweighted_graph_infos,
        };

        // --- 2. WEIGHTED PIPELINE ---
        let mut hg = DatasetLoader::builder()
            .cached(true)
            .$name()
            .weighted()
            .load()?;

        hg.retain_orders(vec![2, 3]);

        let n_w = hg.n();
        let count2_w = hg.0.edges::<2>().len();
        let count3_w = hg.0.edges::<3>().len();
        let count4_w = hg.0.edges::<4>().len(); // Will be 0 due to retain_orders([2, 3])

        let (hyper_adj_w, _, _) =
            HyperAdjList::<()>::from_hypergraph_mapped(hg.0.to_unweighted().clone());
        let mut max_degree_w = 0;
        for neighbors in hyper_adj_w.iter_all_incident_edges() {
            max_degree_w = max_degree_w.max(neighbors.len());
        }
        let (_, _, hyper_degeneracy_w) = hyper_degeneracy_ordering(&hyper_adj_w);

        let edges_w =
            hg.0.edges::<2>()
                .iter()
                .map(|e| (e.nodes[0], e.nodes[1], ()))
                .collect::<Vec<_>>();
        let (adj_w, _, _) = AdjList::<(), Undirected, WithoutIncidence>::from_edges_mapped(edges_w);
        let (_, _, degeneracy_w) = degeneracy_ordering(&adj_w);

        let weighted_graph_infos = GraphInfos {
            n: n_w,
            e2: count2_w,
            e3: count3_w,
            e4: count4_w,
            density2: k_uniform_density(n_w, count2_w, 2).unwrap_or(0.0),
            density3: k_uniform_density(n_w, count3_w, 3).unwrap_or(0.0),
            density4: k_uniform_density(n_w, count4_w, 4).unwrap_or(0.0),
            max_degree: max_degree_w,
            degeneracy: degeneracy_w,
            hyper_degeneracy: hyper_degeneracy_w,
        };

        // Benchmark weighted 3
        let time = Instant::now();
        escape::weighted_3(&hg);
        let weighted_order3 = ExecutionInfos {
            time: time.elapsed(),
            graph_infos: weighted_graph_infos,
        };

        let perf = BenchmarkResult {
            dataset_name: stringify!($name).to_string(),
            description: "".to_string(),
            unweighted: OrderResult {
                order3: unweighted_order3,
                order4,
            },
            weighted: OrderResult {
                order3: weighted_order3,
                order4: ExecutionInfos::default(),
            },
        };

        $rv.push(perf);
    };
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let mut result = Vec::new();

    // Run benchmarks
    test_dataset!(hospital, result);
    test_dataset!(conference, result);
    test_dataset!(dblp, result);
    test_dataset!(enron, result);
    test_dataset!(eu, result);
    test_dataset!(geology, result);
    test_dataset!(high_school, result);
    test_dataset!(history, result);
    test_dataset!(justice, result);
    test_dataset!(ndc_classes, result);
    test_dataset!(ndc_substances, result);
    test_dataset!(pacs, result);
    test_dataset!(primary_school, result);
    test_dataset!(wiki, result);
    test_dataset!(workspace, result);
    test_dataset!(friendship_hs, result);

    // --- LaTeX Table 1: Benchmark Results ---
    println!("\n% --- LaTeX Table Output: Timings ---");
    println!("% Add \\usepackage{{booktabs}} and \\usepackage{{siunitx}} to your LaTeX preamble.");
    println!("\\begin{{table}}[h]");
    println!("    \\centering");
    println!(
        "    \\begin{{tabular}}{{l S[table-format=3.2] S[table-format=3.2] S[table-format=4.2] c}}"
    );
    println!("        \\toprule");
    println!(
        "        & \\multicolumn{{2}}{{c}}{{Order 3 motifs}} & \\multicolumn{{2}}{{c}}{{Order 4 motifs}} \\\\"
    );
    println!("        \\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}");
    println!(
        "        Dataset & {{unweighted}} & {{weighted}} & {{unweighted}} & {{weighted}} \\\\"
    );
    println!("        \\midrule");

    let format_duration_ms = |d: Duration| -> String {
        if d.is_zero() {
            "{---}".to_string()
        } else {
            format!("{:.2}", d.as_secs_f64() * 1000.0)
        }
    };

    for r in &result {
        let uw3 = format_duration_ms(r.unweighted.order3.time);
        let w3 = format_duration_ms(r.weighted.order3.time);
        let uw4 = format_duration_ms(r.unweighted.order4.time);
        let w4 = format_duration_ms(r.weighted.order4.time);

        println!(
            "        \\verb|{:<18}| & {:<15} & {:<15} & {:<15} & {} \\\\",
            r.dataset_name, uw3, w3, uw4, w4
        );
    }

    println!("        \\bottomrule");
    println!("    \\end{{tabular}}");
    println!("    \\caption{{Algorithm execution times by order, in milliseconds (ms).}}");
    println!("    \\label{{tab:benchmark_results}}");
    println!("\\end{{table}}");

    // --- LaTeX Table 2: Dataset Specifications (No Densities) ---
    println!("\n% --- LaTeX Table Output: Dataset Structural Specifications ---");
    println!("\\begin{{table}}[h]");
    println!("    \\centering");
    println!(
        "    \\begin{{tabular}}{{l S[table-format=6.0] S[table-format=6.0] S[table-format=6.0] S[table-format=6.0] S[table-format=5.0] S[table-format=4.0] S[table-format=4.0]}}"
    );
    println!("        \\toprule");
    println!(
        "        Dataset & {{$n$}} & {{$e_2$}} & {{$e_3$}} & {{$e_4$}} & {{max deg}} & {{degen}} & {{hyper degen}} \\\\"
    );
    println!("        \\midrule");

    for r in &result {
        let info = &r.unweighted.order3.graph_infos;
        println!(
            "        \\verb|{:<18}| & {:<8} & {:<8} & {:<8} & {:<8} & {:<8} & {:<8} & {} \\\\",
            r.dataset_name,
            info.n,
            info.e2,
            info.e3,
            info.e4,
            info.max_degree,
            info.degeneracy,
            info.hyper_degeneracy
        );
    }

    println!("        \\bottomrule");
    println!("    \\end{{tabular}}");
    println!("    \\caption{{Structural properties of the benchmarked datasets.}}");
    println!("    \\label{{tab:dataset_specs_structural}}");
    println!("\\end{{table}}");

    // --- LaTeX Table 3: Dataset Densities ---
    println!("\n% --- LaTeX Table Output: Dataset Densities ---");
    println!("\\begin{{table}}[h]");
    println!("    \\centering");
    println!(
        "    \\begin{{tabular}}{{l S[table-format=1.2e-2] S[table-format=1.2e-2] S[table-format=1.2e-2]}}"
    );
    println!("        \\toprule");
    println!(
        "        Dataset & {{density\\textsubscript{{2}}}} & {{density\\textsubscript{{3}}}} & {{density\\textsubscript{{4}}}} \\\\"
    );
    println!("        \\midrule");

    for r in &result {
        let info = &r.unweighted.order3.graph_infos;
        println!(
            "        \\verb|{:<18}| & {:.3e} & {:.3e} & {:.3e} \\\\",
            r.dataset_name, info.density2, info.density3, info.density4
        );
    }

    println!("        \\bottomrule");
    println!("    \\end{{tabular}}");
    println!("    \\caption{{Density properties of the benchmarked datasets across orders.}}");
    println!("    \\label{{tab:dataset_specs_densities}}");
    println!("\\end{{table}}");

    Ok(())
}
