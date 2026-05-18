use polars::lazy::prelude::*;
use polars::prelude::*;

use seq_macro::seq;
use std::{error::Error, path::Path};

use crate::{
    graph::{Hx, Hypergraph, NodeId, NodeWeight},
    loader::common::Loader,
};

// Use polars for robust CSV parsing and grouping similar to the Python version.
pub fn load_pacs_common_from_csv<P>(
    dataset_location: &P,
) -> Result<Vec<(String, Vec<NodeId>)>, Box<dyn Error>>
where
    P: AsRef<Path> + ?Sized,
{
    let path = dataset_location.as_ref();

    // Read the CSV into a DataFrame, scanning whole file for schema so ArticleID isn't inferred as numeric.
    // In Polars 0.52, infer_schema_length=0 forces string columns.
    //
    //

    let q = LazyCsvReader::new("docs/assets/data/iris.csv")
        .with_has_header(true)
        .finish()?
        .filter(col("sepal_length").gt(lit(5)))
        .group_by(vec![col("species")])
        .agg([col("sepal_width").mean()]);
    let df = q.collect()?;

    // let lf = CsvReadOptions::default()
    //     .with_has_header(true)
    //     .with_infer_schema_length(Some(0))
    //     .with_columns(Some(Arc::new(["ArticleID".into(), "FullName".into()])))
    //     .try_into_reader_with_file_path(Some(path.into()))?
    //     .finish()?;

    // let names_lf = LazyFrame::select([col("FullName")])
    //     .filter(col("FullName").is_not_null())
    //     .unique(
    //         Some(vec!["FullName".to_string()]), // columns to de-duplicate on
    //         UniqueKeepStrategy::First,          // keep strategy
    //     )
    //     // with_row_index takes (name, offset)
    //     .with_row_index("author_id", None);

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(Some(0))
        .with_columns(Some(Arc::new(["ArticleID".into(), "FullName".into()])))
        .try_into_reader_with_file_path(Some(path.into()))?
        .finish()?;

    // In Polars 0.52 strings are String dtype (previously Utf8).
    let article_col = df.column("ArticleID")?.str()?;
    let fullname_col = df.column("FullName")?.str()?;

    // Build global name->id mapping (first-seen order) and per-article lists of ids preserving order
    use std::collections::HashMap;
    let mut name2id: HashMap<String, NodeId> = HashMap::new();
    let mut next_id: NodeId = 0;

    let mut article_index: HashMap<String, usize> = HashMap::new();
    let mut articles: Vec<(String, Vec<NodeId>)> = Vec::new();

    let height = df.height();
    for i in 0..height {
        if let (Some(article), Some(fullname)) = (article_col.get(i), fullname_col.get(i)) {
            let article = article.trim();
            let fullname = fullname.trim();
            if article.is_empty() || fullname.is_empty() {
                continue;
            }

            let id = *name2id.entry(fullname.to_string()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });

            let idx = *article_index.entry(article.to_string()).or_insert_with(|| {
                let idx = articles.len();
                articles.push((article.to_string(), Vec::new()));
                idx
            });

            let entry = &mut articles[idx].1;
            if !entry.contains(&id) {
                entry.push(id);
            }
        }
    }

    Ok(articles)
}

pub struct Unweighted;
pub struct Weighted;

impl Loader for Unweighted {
    const NAME: &'static str = "UW_PACS";
    type Output = Hypergraph<NodeId, ()>;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
        let tuples = load_pacs_common_from_csv(dataset_location)?;
        let mut hg = Hypergraph::new();

        for (_article, ids) in tuples.into_iter() {
            if ids.len() > 1 && ids.len() <= 10 {
                // bucket by size using seq_macro
                seq!(N in 2..11 {
                    if ids.len() == N {
                        let mut arr = [0 as NodeId; N];
                        for i in 0..N { arr[i] = ids[i]; }
                        // create Hx via Hx::new_unchecked for unweighted
                        hg.add_edge(Hx::new_unchecked(arr, ()));
                    }
                });
            }
        }

        Ok(hg)
    }
}

impl Loader for Weighted {
    const NAME: &'static str = "W_PACS";
    type Output = Hypergraph<NodeId, NodeWeight>;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
        let tuples = load_pacs_common_from_csv(dataset_location)?;
        let mut hg = Hypergraph::new();

        for (_article, ids) in tuples.into_iter() {
            if ids.len() > 1 && ids.len() <= 10 {
                seq!(N in 2..11 {
                    if ids.len() == N {
                        let mut arr = [0 as NodeId; N];
                        for i in 0..N { arr[i] = ids[i]; }

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
