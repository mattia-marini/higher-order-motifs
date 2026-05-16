use ouroboros::self_referencing;
use rkyv::bytecheck::CheckBytes;
use rkyv::de::Pool;
use rkyv::rancor::Strategy;
use rkyv::util::AlignedVec;
use rkyv::validation::Validator;
use rkyv::validation::archive::ArchiveValidator;
use rkyv::validation::shared::SharedValidator;
use rkyv::{Archive, Deserialize};
use std::cmp::Eq;
use std::fs::File;
use std::hash::Hash;
use std::path::Path;
use std::{error::Error, io::Write};

use crate::graph::{ArchivedHx, ArchivedHypergraph, Hx, Hypergraph};

pub trait StdSerializable:
    for<'a> rkyv::Serialize<
        rkyv::rancor::Strategy<
            rkyv::ser::Serializer<
                rkyv::util::AlignedVec,
                rkyv::ser::allocator::ArenaHandle<'a>,
                rkyv::ser::sharing::Share,
            >,
            rkyv::rancor::Error,
        >,
    >
{
}
impl<T> StdSerializable for T where
    T: for<'a> rkyv::Serialize<
            rkyv::rancor::Strategy<
                rkyv::ser::Serializer<
                    rkyv::util::AlignedVec,
                    rkyv::ser::allocator::ArenaHandle<'a>,
                    rkyv::ser::sharing::Share,
                >,
                rkyv::rancor::Error,
            >,
        >
{
}

pub trait StdDeserializable<T>: Deserialize<T, Strategy<Pool, rkyv::rancor::Error>> {}
impl<T, U> StdDeserializable<T> for U where U: Deserialize<T, Strategy<Pool, rkyv::rancor::Error>> {}

pub trait StdCheckBytes<'a>:
    CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rkyv::rancor::Error>>
{
}
impl<'a, T> StdCheckBytes<'a> for T where
    T: CheckBytes<Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rkyv::rancor::Error>>
{
}

impl<T, W> Hypergraph<T, W>
where
    T: Archive,
    W: Archive,
{
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>>
    where
        T: StdSerializable + Hash + Eq,
        <T as Archive>::Archived: Hash + Eq,
        W: StdSerializable,
    {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self)?;

        let mut file = File::create(path)?;
        file.write_all(&bytes)?;

        Ok(())
    }

    pub fn load_deserialized<P: AsRef<Path>>(path: P) -> Result<Hypergraph<T, W>, Box<dyn Error>>
    where
        for<'a> <T as Archive>::Archived: StdCheckBytes<'a>,
        <T as Archive>::Archived: StdDeserializable<T>,
        <T as Archive>::Archived: Hash + Eq,
        T: Hash + Eq,
        for<'a> <W as Archive>::Archived: StdCheckBytes<'a>,
        <W as Archive>::Archived: StdDeserializable<W>,
    {
        let mut file = std::fs::File::open(path)?;
        let mut bytes: AlignedVec = rkyv::util::AlignedVec::new();
        bytes.extend_from_reader(&mut file)?;

        // let start = std::time::Instant::now();
        let archived = rkyv::access::<ArchivedHypergraph<T, W>, rkyv::rancor::Error>(&bytes[..])?;
        // println!("Loading ArchivedHypergraph {:?}", start.elapsed());

        // let start = std::time::Instant::now();
        let mut rv = rkyv::deserialize::<Hypergraph<T, W>, rkyv::rancor::Error>(archived)?;
        // println!("Loading Hypergraph{:?}", start.elapsed());

        Ok(rv)
    }

    pub fn load_archived<P: AsRef<Path>>(
        path: P,
    ) -> Result<ArchivedHypergraphHandle<T, W>, Box<dyn Error>>
    where
        for<'a> <T as Archive>::Archived: StdCheckBytes<'a>,
        for<'a> <W as Archive>::Archived: StdCheckBytes<'a>,
        <T as Archive>::Archived: Hash + Eq,
        T: Hash + Eq,
    {
        let mut file = std::fs::File::open(path)?;
        let mut bytes: AlignedVec = rkyv::util::AlignedVec::new();
        bytes.extend_from_reader(&mut file)?;

        let archived = ArchivedHypergraphHandle::<T, W>::try_new(bytes, |b| {
            rkyv::access::<ArchivedHypergraph<T, W>, rkyv::rancor::Error>(&b[..])
        })?;
        Ok(archived)
    }
}

impl<const N: usize, T, W> Hash for ArchivedHx<N, T, W>
where
    T: Hash + Archive,
    W: Archive,
    <T as Archive>::Archived: Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.nodes.hash(state);
    }
}

impl<const N: usize, T, W> PartialEq for ArchivedHx<N, T, W>
where
    T: Archive,
    <T as Archive>::Archived: PartialEq,
    W: Archive,
{
    fn eq(&self, other: &Self) -> bool {
        self.nodes == other.nodes
    }
}

impl<const N: usize, T, W> Eq for ArchivedHx<N, T, W>
where
    T: Archive,
    <T as Archive>::Archived: Eq,
    W: Archive,
{
}

#[self_referencing]
pub struct ArchivedHypergraphHandle<T, W>
where
    T: Archive + Hash + Eq + 'static,
    W: Archive + 'static,
    <T as Archive>::Archived: Hash + Eq,
{
    bytes: AlignedVec,
    #[borrows(bytes)]
    pub archived: &'this super::ArchivedHypergraph<T, W>,
}
