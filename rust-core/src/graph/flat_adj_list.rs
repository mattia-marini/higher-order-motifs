use num_traits::{AsPrimitive, One, PrimInt, Unsigned, Zero};

use rkyv::{
    Archive, Deserialize, Serialize,
    bytecheck::CheckBytes,
    de::Pool,
    deserialize,
    rancor::Strategy,
    validation::{Validator, archive::ArchiveValidator, shared::SharedValidator},
};

use std::{
    fs::File,
    hash::Hash,
    io::Write,
    ops::{AddAssign, Index},
    path::Path,
};

use super::adj_list::AdjList;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
pub struct FlatAdjList<T, E> {
    pub offsets: Vec<T>,
    pub edges: Vec<E>,
}

impl<T, E> Index<usize> for FlatAdjList<T, E>
where
    T: AsPrimitive<usize>,
{
    type Output = [E];

    fn index(&self, index: usize) -> &Self::Output {
        let start = self.offsets[index].as_();
        let end = self.offsets[index + 1].as_();
        &self.edges[start..end]
    }
}

impl<T, E> FlatAdjList<T, E>
where
    T: AsPrimitive<usize> + num_traits::Zero,
    E: AsPrimitive<usize> + num_traits::Zero + Clone + Ord + AddAssign + Hash + Eq + One,
    usize: AsPrimitive<E>,
    usize: AsPrimitive<T>,
{
    pub fn new() -> Self {
        Self {
            offsets: vec![T::zero()],
            edges: vec![],
        }
    }

    pub fn from_edges(edges: &[(E, E)], _directed: bool) -> Self {
        let mut adj_list = AdjList::<E>::from_edges(edges);
        adj_list.remove_self_loops();
        // adj_list.remove_multiedges();
        adj_list.make_undirected();

        let mut rv = Self {
            offsets: vec![T::zero(); adj_list.n() + 1],
            edges: vec![E::zero(); adj_list.m()],
        };

        println!("Constructed adjacency list. n: {}, m: {}", rv.n(), rv.m());

        for (i, neighbors) in adj_list.adj.iter().enumerate() {
            rv.offsets[i + 1] = rv.offsets[i] + neighbors.len().as_();
            rv.edges[rv.offsets[i].as_()..rv.offsets[i + 1].as_()].copy_from_slice(neighbors);
        }
        println!(
            "Constructed flat adjacency list. n: {}, m: {}",
            rv.n(),
            rv.m()
        );

        rv
    }

    pub fn n(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn m(&self) -> usize {
        self.edges.len()
    }
}

impl<T, E> FlatAdjList<T, E>
where
    T: for<'a> rkyv::Serialize<
            rkyv::rancor::Strategy<
                rkyv::ser::Serializer<
                    rkyv::util::AlignedVec,
                    rkyv::ser::allocator::ArenaHandle<'a>,
                    rkyv::ser::sharing::Share,
                >,
                rkyv::rancor::Error,
            >,
        >,
    E: for<'a> rkyv::Serialize<
            rkyv::rancor::Strategy<
                rkyv::ser::Serializer<
                    rkyv::util::AlignedVec,
                    rkyv::ser::allocator::ArenaHandle<'a>,
                    rkyv::ser::sharing::Share,
                >,
                rkyv::rancor::Error,
            >,
        >,
{
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self)?;
        // .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let mut file = File::create(path)?;
        file.write_all(&bytes)?;

        Ok(())
    }
}

impl<T, E> FlatAdjList<T, E>
where
    T: Archive,
    E: Archive,
    for<'a> <T as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rkyv::rancor::Error>>,
    for<'a> <E as Archive>::Archived:
        CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rkyv::rancor::Error>>,
    ArchivedFlatAdjList<T, E>: Deserialize<FlatAdjList<T, E>, Strategy<Pool, rkyv::rancor::Error>>,
{
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let archived =
            rkyv::access::<ArchivedFlatAdjList<T, E>, rkyv::rancor::Error>(&bytes[..]).unwrap();

        let rv = deserialize::<FlatAdjList<T, E>, rkyv::rancor::Error>(archived)?;
        Ok(rv)
    }
}
