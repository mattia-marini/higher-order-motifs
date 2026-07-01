use rust_core_macros::hoist_mod;
use std::fmt::{self, Display};
use thiserror::Error;

use super::NodeId;
use super::hypergraph::edge_collection::{MAX_HX_SIZE, MIN_HX_SIZE};

#[derive(Error, Debug)]
pub enum HypergraphError<T> {
    // thiserror is smart enough to only require T: Display
    // for the generated impl Display for GraphError<T>
    #[error("Hyperedges cannot have duplicate nodes: {0}")]
    DuplicateNodes(T),

    #[error("Unexpected hyperedge size: expected {expected}, got {got}")]
    InvalidHyperedgeSize { expected: T, got: T },

    #[error("Hyperedges supports orders {MAX_HX_SIZE} to {MIN_HX_SIZE}; got hyperedge of size {0}")]
    UnsupportedHyperedgeSize(T),

    #[error("Generic error: {0}")]
    Unknown(String),
}

#[cfg(feature = "bindings")]
#[hoist_mod]
mod bindings {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    impl<T> From<HypergraphError<T>> for PyErr
    where
        T: Display,
    {
        fn from(err: HypergraphError<T>) -> PyErr {
            PyValueError::new_err(err.to_string())
        }
    }
}

// impl<T> From<PyErr> for GraphError<T> {
//     fn from(err: PyErr) -> Self {
//         GraphError::GenericError(format!("Python error: {}", err))
//     }
// }
