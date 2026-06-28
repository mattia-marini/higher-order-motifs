use polars::lazy::prelude::*;
use polars::prelude::*;

use seq_macro::seq;
use std::{error::Error, path::Path, sync::Arc};

use crate::{
    graph::{Hx, Hypergraph, NodeId, NodeWeight, UnweightedHypergraph, WeightedHypergraph},
    loader::common::Loader,
    loader::error::LoaderError,
};

pub fn load_pacs_common_from_csv<P>(dataset_location: &P) -> Result<Vec<Vec<NodeId>>, LoaderError>
where
    P: AsRef<Path> + ?Sized,
{
    let path = dataset_location;

    let mut schema = Schema::with_capacity(4);
    schema.insert("ArticleID".into(), DataType::String);
    schema.insert("PACS".into(), DataType::UInt32);
    schema.insert("AuthorDAIS".into(), DataType::UInt32);
    schema.insert("FullName".into(), DataType::String);

    let lf = LazyCsvReader::new(path)
        .with_null_values(Some(NullValues::AllColumnsSingle("".into())))
        .with_schema(Some(Arc::new(schema)))
        .finish()
        .map_err(|e| LoaderError::Unknown(format!("Failed to read csv: {}", e)))?;

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
        .agg([col("author_id").unique().alias("authors")]);

    let rv = grouped_lf
        .collect()
        .map_err(|e| LoaderError::Unknown(format!("Polars error: {}", e)))?;

    let authors_col = rv
        .column("authors")
        .map_err(|e| LoaderError::Unknown(format!("Column error: {}", e)))?
        .list()
        .map_err(|e| LoaderError::Unknown(format!("List error: {}", e)))?;

    Ok(authors_col
        .into_iter()
        .filter_map(|opt_auth_series| {
            let series = opt_auth_series?;

            let mut author_ids: Vec<NodeId> = series
                .u32()
                .expect("Authors inner list was not a UInt32 type")
                .into_iter()
                .flatten()
                .map(|val| val as NodeId)
                .collect();

            if author_ids.len() > 1 && author_ids.len() <= 10 {
                author_ids.sort_unstable();
                Some(author_ids)
            } else {
                None
            }
        })
        .collect())
}

use super::{PacsStdUnweightedLoader, PacsStdWeightedLoader};

pub struct Unweighted;
pub struct Weighted;

impl Loader for PacsStdUnweightedLoader {
    type Output = UnweightedHypergraph;

    const VARIANT: &'static str = "uw";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        let valid_author_groups = load_pacs_common_from_csv(&dataset_location)?;
        let mut hg = Hypergraph::new();

        for ids in valid_author_groups {
            seq!(N in 2..11 {
                if ids.len() == N {
                    let mut arr = [0 as NodeId; N];
                    arr.copy_from_slice(&ids);
                    hg.add_edge(Hx::new(arr, ()).unwrap());
                }
            });
        }

        Ok(hg.into())
    }
}

impl Loader for PacsStdWeightedLoader {
    type Output = WeightedHypergraph;

    const VARIANT: &'static str = "w";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        let valid_author_groups = load_pacs_common_from_csv(&dataset_location)?;
        let mut hg = Hypergraph::new();

        for ids in valid_author_groups {
            seq!(N in 2..11 {
                if ids.len() == N {
                    let mut arr = [0 as NodeId; N];
                    arr.copy_from_slice(&ids);
                    if !hg.has_hyperedge(&arr) {
                        hg.add_edge(Hx::new_unchecked(arr, 0.0));
                    }
                    hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                }
            });
        }

        Ok(hg.into())
    }
}
