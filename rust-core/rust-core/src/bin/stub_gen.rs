use rust_core::stub_info;

use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub_infos = stub_info()?;
    stub_infos.generate()?;
    Ok(())
}
