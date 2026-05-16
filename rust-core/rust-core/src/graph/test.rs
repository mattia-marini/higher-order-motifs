use pyo3::{pyclass, pymethods};
use rust_core_macros::inherent;

trait X {
    fn f();
}

#[pyclass]
struct Y {}

#[inherent(attr(pymethods))]
impl X for Y {
    #[inner(attr(staticmethod))]
    pub fn f() {
        println!("Hello from X::f!");
    }
}
