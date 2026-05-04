pub trait StaticHypergraph {
    fn n(&self) -> usize;
    fn m(&self) -> usize;
    fn has_edge(&self, edge: &Hx) -> bool;
    fn has_big_edge(&self, edge: &Hx) -> bool;
    fn has_h2(&self, edge: &H2) -> bool;
    fn has_h3(&self, edge: &H3) -> bool;
    fn has_h4(&self, edge: &H4) -> bool;
    fn has_h5(&self, edge: &H5) -> bool;
}

/// All operations that require mutable access to the hypergraph should be defined in this trait
pub trait LiveHypergraph {}

#[derive(FromPyObject, Debug)]
pub enum Hypergraph<'py> {
    #[pyo3(transparent)]
    Db(Bound<'py, UnweightedHypergraph>),
    #[pyo3(transparent)]
    File(Bound<'py, ArchivedUnweightedHypergraph>),
}

impl_stub_type!(Hypergraph<'_> = UnweightedHypergraph | ArchivedUnweightedHypergraph);
