use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use hashbrown::HashMap;
use seq_macro::seq;

use crate::{
    graph::{Hx, Hypergraph, NodeId, NodeWeight},
    loader::common::Loader,
};

pub struct Unweighted;
pub struct Weighted;

use polars::prelude::*;
use std::sync::Arc;

impl Loader for Unweighted {
    const NAME: &'static str = "UW_justice";
    type Output = Hypergraph<NodeId, ()>;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
        let path = dataset_location.as_ref();

        // 1. Read the CSV lazily without headers to access columns by their index.
        // Polars automatically names them "column_1", "column_2", etc.
        let lf = LazyCsvReader::new(path).with_ignore_errors(true).finish()?;

        // 2. Select and cast only the columns we need (indices 0, 54, 55 map to 1, 55, 56)
        let lf = lf
            .select([
                col("caseId").alias("case_id"),
                col("justiceName").alias("justice_name"),
                col("vote").cast(DataType::Int32).alias("vote"),
            ])
            // Drop rows where vote failed to parse, or where core fields are missing
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
            // Grouping by both gives us the exact equivalent of your nested HashMaps
            .group_by([col("case_id"), col("vote")])
            .agg([col("justice_id").unique().alias("justices")]);

        let df = grouped_lf.collect()?;

        // 5. Build the Hypergraph
        let mut hg = Hypergraph::new();
        let justices_col = df.column("justices")?.list()?;

        for opt_justice_series in justices_col.into_iter() {
            let Some(series) = opt_justice_series else {
                continue;
            };

            // Polars row indices are generated as u32 (UInt32Chunked)
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

        Ok(hg)
    }
}

impl Loader for Weighted {
    const NAME: &'static str = "W_justice";
    type Output = Hypergraph<NodeId, NodeWeight>;

    fn from_file<P>(dataset_location: &P) -> Result<Self::Output, Box<dyn Error>>
    where
        P: AsRef<Path> + ?Sized,
    {
        let path = dataset_location.as_ref();

        // 1. Read the CSV lazily without headers to access columns by their index.
        // Polars automatically names them "column_1", "column_2", etc.
        let lf = LazyCsvReader::new(path).with_ignore_errors(true).finish()?;

        // 2. Select and cast only the columns we need (indices 0, 54, 55 map to 1, 55, 56)
        let lf = lf
            .select([
                col("caseId").alias("case_id"),
                col("justiceName").alias("justice_name"),
                col("vote").cast(DataType::Int32).alias("vote"),
            ])
            // Drop rows where vote failed to parse, or where core fields are missing
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
            // Grouping by both gives us the exact equivalent of your nested HashMaps
            .group_by([col("case_id"), col("vote")])
            .agg([col("justice_id").unique().alias("justices")]);

        let df = grouped_lf.collect()?;

        // 5. Build the Hypergraph
        let mut hg = Hypergraph::new();
        let justices_col = df.column("justices")?.list()?;

        for opt_justice_series in justices_col.into_iter() {
            let Some(series) = opt_justice_series else {
                continue;
            };

            // Polars row indices are generated as u32 (UInt32Chunked)
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

        Ok(hg)
    }
}
