use rust_core_macros::hoist_mod;
use std::{
    fmt::{self, Display},
    path::PathBuf,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("Io error occurred while accessing path: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Malformed dataset file: {0}")]
    MlformedDataset(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

#[cfg(feature = "bindings")]
#[hoist_mod]
mod bindings {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    impl From<LoaderError> for PyErr {
        fn from(err: LoaderError) -> PyErr {
            PyValueError::new_err(err.to_string())
        }
    }
}
