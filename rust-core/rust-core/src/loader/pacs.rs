use polars::lazy::prelude::*;
use polars::prelude::*;

use seq_macro::seq;
use std::{error::Error, path::Path};

use crate::{
    graph::{Hx, Hypergraph, NodeId, NodeWeight},
    loader::common::Loader,
};

pub fn load_pacs_common_from_csv<P>(
    dataset_location: &P,
) -> Result<Vec<Vec<NodeId>>, Box<dyn Error>>
where
    P: AsRef<Path> + ?Sized,
{
    let path = dataset_location.as_ref();

    let mut schema = Schema::with_capacity(4);
    schema.insert("ArticleID".into(), DataType::String);
    schema.insert("PACS".into(), DataType::UInt32);
    schema.insert("AuthorDAIS".into(), DataType::UInt32);
    schema.insert("FullName".into(), DataType::String);

    let lf = LazyCsvReader::new(path)
        .with_null_values(Some(NullValues::AllColumnsSingle("".into())))
        .with_schema(Some(Arc::new(schema)))
        .finish()?;

    let names_lf = lf
        .clone()
        .select([col("FullName")])
        .filter(col("FullName").is_not_null())
        .unique(
            Some(vec!["FullName".to_string()]),
            UniqueKeepStrategy::First,
        )
        .with_row_index("author_id", None);

    let grouped_lf = lf
        .join_builder()
        .with(names_lf)
        .on([col("FullName")])
        .how(JoinType::Left)
        .finish()
        .group_by([col("ArticleID")])
        .agg([
            col("author_id").unique().alias("authors"),
            // Removed PACS since it doesn't seem to be used in the graph!
        ]);

    let rv = grouped_lf.collect()?;

    let authors_col = rv.column("authors")?.list()?;

    // Use filter_map to avoid extracting data we don't need
    Ok(authors_col
        .into_iter()
        .filter_map(|opt_auth_series| {
            let series = opt_auth_series?; // Returns None (skips) if null

            let mut author_ids: Vec<NodeId> = series
                .u32()
                .expect("Authors inner list was not a UInt32 type")
                .into_iter()
                .flatten()
                .map(|val| val as NodeId)
                .collect();

            // Filter sizes *before* returning them to save memory
            if author_ids.len() > 1 && author_ids.len() <= 10 {
                // Highly recommended: Sort so [A, B] and [B, A] are identical edges
                author_ids.sort_unstable();
                Some(author_ids)
            } else {
                None
            }
        })
        .collect())
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
        let valid_author_groups = load_pacs_common_from_csv(dataset_location)?;
        let mut hg = Hypergraph::new();

        for ids in valid_author_groups {
            // We already filtered lengths 2..=10 in the loader!
            seq!(N in 2..11 {
                if ids.len() == N {
                    let mut arr = [0 as NodeId; N];
                    arr.copy_from_slice(&ids); // Blazing fast memory copy
                    hg.add_edge(Hx::new(arr, ()).unwrap());
                }
            });
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
        let valid_author_groups = load_pacs_common_from_csv(dataset_location)?;
        let mut hg = Hypergraph::new();

        for ids in valid_author_groups {
            seq!(N in 2..11 {
                if ids.len() == N {
                    let mut arr = [0 as NodeId; N];
                    arr.copy_from_slice(&ids); // Blazing fast memory copy

                    // If your graph API has an `add_or_modify` or `entry` API,
                    // use that to avoid hashing `arr` twice!
                    if !hg.has_hyperedge(&arr) {
                        hg.add_edge(Hx::new_unchecked(arr, 0.0));
                    }
                    hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                }
            });
        }

        Ok(hg)
    }
}
