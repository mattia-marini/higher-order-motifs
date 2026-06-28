use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use hashbrown::HashMap;
use seq_macro::seq;

use crate::{
    graph::{Hx, Hypergraph, NodeId, NodeWeight, UnweightedHypergraph, WeightedHypergraph},
    loader::common::Loader,
    loader::error::LoaderError,
};

pub struct Unweighted;
pub struct Weighted;

use polars::prelude::*;
use std::sync::Arc;

use super::{JusticeStdUnweightedLoader, JusticeStdWeightedLoader};

impl Loader for JusticeStdUnweightedLoader {
    type Output = UnweightedHypergraph;

    const VARIANT: &'static str = "uw";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        let path = dataset_location;

        // 1. Read the CSV lazily without headers to access columns by their index.
        let lf = LazyCsvReader::new(path)
            .with_ignore_errors(true)
            .finish()
            .map_err(|e| LoaderError::Unknown(format!("Polars reader error: {}", e)))?;

        // 2. Select and cast only the columns we need
        let lf = lf
            .select([
                col("caseId").alias("case_id"),
                col("justiceName").alias("justice_name"),
                col("vote").cast(DataType::Int32).alias("vote"),
            ])
            .filter(col("vote").is_not_null())
            .filter(col("justice_name").is_not_null())
            .filter(col("case_id").is_not_null());

        // 3. Extract unique justices and assign a NodeId (row_index)
        let names_lf = lf
            .clone()
            .select([col("justice_name")])
            .unique(
                Some(vec!["justice_name".to_string()]),
                UniqueKeepStrategy::First,
            )
            .with_row_index("justice_id", None);

        // 4. Join the NodeIds back, group by Case AND Vote, and collect the justices
        let grouped_lf = lf
            .join_builder()
            .with(names_lf)
            .on([col("justice_name")])
            .how(JoinType::Left)
            .finish()
            .group_by([col("case_id"), col("vote")])
            .agg([col("justice_id").unique().alias("justices")]);

        let df = grouped_lf
            .collect()
            .map_err(|e| LoaderError::Unknown(format!("Polars collect error: {}", e)))?;

        // 5. Build the Hypergraph
        let mut hg = Hypergraph::new();
        let justices_col = df
            .column("justices")
            .map_err(|e| LoaderError::Unknown(format!("Column error: {}", e)))?
            .list()
            .map_err(|e| LoaderError::Unknown(format!("List error: {}", e)))?;

        for opt_justice_series in justices_col.into_iter() {
            let Some(series) = opt_justice_series else {
                continue;
            };

            let mut justice_ids: Vec<NodeId> = series
                .u32()
                .expect("Justices inner list was not a UInt32 type")
                .into_iter()
                .flatten()
                .map(|val| val as NodeId)
                .collect();

            justice_ids.sort_unstable();
            justice_ids.dedup();

            let len = justice_ids.len();

            seq!(N in 2..11 {
                match len {
                    #(
                        N => {
                            let mut arr =  [0 as NodeId; N];
                            arr.copy_from_slice(&justice_ids);
                            hg.add_edge(Hx::new_unchecked(arr, ()));
                        },
                    )*
                    _ => ()
                }
            });
        }

        Ok(hg.into())
    }
}

impl Loader for JusticeStdWeightedLoader {
    type Output = WeightedHypergraph;

    const VARIANT: &'static str = "w";

    fn from_file(&self) -> Result<Self::Output, LoaderError> {
        let dataset_location = self.dataset_location.clone();
        let path = dataset_location;

        // 1. Read the CSV lazily without headers to access columns by their index.
        let lf = LazyCsvReader::new(path)
            .with_ignore_errors(true)
            .finish()
            .map_err(|e| LoaderError::Unknown(format!("Polars reader error: {}", e)))?;

        // 2. Select and cast only the columns we need
        let lf = lf
            .select([
                col("caseId").alias("case_id"),
                col("justiceName").alias("justice_name"),
                col("vote").cast(DataType::Int32).alias("vote"),
            ])
            .filter(col("vote").is_not_null())
            .filter(col("justice_name").is_not_null())
            .filter(col("case_id").is_not_null());

        // 3. Extract unique justices and assign a NodeId (row_index)
        let names_lf = lf
            .clone()
            .select([col("justice_name")])
            .unique(
                Some(vec!["justice_name".to_string()]),
                UniqueKeepStrategy::First,
            )
            .with_row_index("justice_id", None);

        // 4. Join the NodeIds back, group by Case AND Vote, and collect the justices
        let grouped_lf = lf
            .join_builder()
            .with(names_lf)
            .on([col("justice_name")])
            .how(JoinType::Left)
            .finish()
            .group_by([col("case_id"), col("vote")])
            .agg([col("justice_id").unique().alias("justices")]);

        let df = grouped_lf
            .collect()
            .map_err(|e| LoaderError::Unknown(format!("Polars collect error: {}", e)))?;

        // 5. Build the Hypergraph
        let mut hg = Hypergraph::new();
        let justices_col = df
            .column("justices")
            .map_err(|e| LoaderError::Unknown(format!("Column error: {}", e)))?
            .list()
            .map_err(|e| LoaderError::Unknown(format!("List error: {}", e)))?;

        for opt_justice_series in justices_col.into_iter() {
            let Some(series) = opt_justice_series else {
                continue;
            };

            let mut justice_ids: Vec<NodeId> = series
                .u32()
                .expect("Justices inner list was not a UInt32 type")
                .into_iter()
                .flatten()
                .map(|val| val as NodeId)
                .collect();

            justice_ids.sort_unstable();
            justice_ids.dedup();

            let len = justice_ids.len();

            seq!(N in 2..11 {
                match len {
                    #(
                        N => {
                            let mut arr =  [0 as NodeId; N];
                            arr.copy_from_slice(&justice_ids);
                            if !hg.has_hyperedge(&arr) { hg.add_edge(Hx::new_unchecked(arr, 0.0)); }
                            hg.modify_hx_weigth_with(&arr, |w| w + 1.0);
                        },
                    )*
                    _ => ()
                }
            });
        }

        Ok(hg.into())
    }
}
