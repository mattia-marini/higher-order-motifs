use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;
use thiserror::Error;

use super::types::NodeId;

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("H2 cannot have duplicate nodes: {0}")]
    DuplicateNodes(NodeId),

    #[error("H2 cannot have duplicate nodes: expected {expected} , got {got}")]
    InvalidHyperedgeSize { expected: usize, got: usize },
}

impl From<GraphError> for PyErr {
    fn from(err: GraphError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
