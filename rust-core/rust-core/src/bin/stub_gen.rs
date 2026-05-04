use pyo3_stub_gen::{Result, StubInfo};
use rkyv::rancor::Error;

fn main() -> Result<()> {
    // match rust_core::stub_info() {
    //     Ok(stub) => stub.generate()?,
    //     Err(e) => eprintln!("Error generating stubs: {}", e.is),
    // }
    //
    let manifest_dir: &::std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    let manifest_dir = manifest_dir
        .parent()
        .ok_or(Err("Can't find workspace root"))?;

    println!("Manifest directory: {}", manifest_dir.display());
    let stub = StubInfo::from_pyproject_toml(manifest_dir.join("pyproject.toml"))?;
    // StubInfo::from_pyproject_toml(manifest_dir.join("pyproject.toml"))
    // let stub = rust_core::stub_info()?;
    stub.generate()?;
    Ok(())
}
