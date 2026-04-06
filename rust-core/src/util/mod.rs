pub mod submodules_initializer;

#[allow(dead_code)]
pub fn timed_custom_printer(_default_format: &str, f_name: &str, duration: std::time::Duration) {
    println!("[Rust time] {}:  {} s", f_name, duration.as_secs_f32());
}
