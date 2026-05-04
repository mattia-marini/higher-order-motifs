use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;

use super::types::NodeId;

#[derive(Debug)]
pub enum GraphError {
    DuplicateNodes(NodeId),
    InvalidHyperedgeSize(usize, usize),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::DuplicateNodes(u) => {
                write!(f, "H2 cannot have duplicate nodes: {} ", u)
            }
            GraphError::InvalidHyperedgeSize(expected, got) => {
                write!(f, "Wrong node size; expected {}, got{} ", expected, got)
            }
        }
    }
}

impl std::error::Error for GraphError {}

impl From<GraphError> for PyErr {
    fn from(err: GraphError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
