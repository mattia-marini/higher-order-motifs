mod ct_map;
mod hoist_mod;
mod inherent;
mod repeat;
mod loader;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn repeat(attr: TokenStream, item: TokenStream) -> TokenStream {
    repeat::repeat(attr, item)
}

#[proc_macro_attribute]
pub fn hoist_mod(attr: TokenStream, item: TokenStream) -> TokenStream {
    hoist_mod::hoist_mod(attr, item)
}

#[proc_macro_attribute]
pub fn inherent(attr: TokenStream, item: TokenStream) -> TokenStream {
    inherent::inherent(attr, item)
}

#[proc_macro_attribute]
pub fn ct_map(attr: TokenStream, item: TokenStream) -> TokenStream {
    ct_map::ct_map(attr, item)
}

#[proc_macro_attribute]
pub fn ct_map_accessor(attr: TokenStream, item: TokenStream) -> TokenStream {
    ct_map::ct_map_accessor(attr, item)
}

#[proc_macro_attribute]
pub fn loader(attr: TokenStream, item: TokenStream) -> TokenStream {
    loader::loader(attr, item)
}
