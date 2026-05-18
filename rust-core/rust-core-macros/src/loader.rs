use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{FnArg, ItemFn, LitStr, Pat, Result, parse_macro_input};

#[derive(Default)]
struct LoaderArgs {
    cache: Option<String>,
}

struct LoaderArgsFinalized {
    cache: String,
}

fn finalize_args(raw: LoaderArgs, fn_name: &str) -> LoaderArgsFinalized {
    let default_cache = fn_name.strip_prefix("load_").unwrap_or(fn_name).to_string();
    let cache = raw
        .cache
        .unwrap_or_else(|| format!("{}.bin", default_cache));

    LoaderArgsFinalized { cache }
}

/// The macro entry point
pub fn loader(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut raw_args = LoaderArgs::default();

    // 1. Idiomatic syn 2.0 closure-based parser for attribute arguments
    let args_parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("cache") {
            if raw_args.cache.is_some() {
                return Err(meta.error("duplicate `cache` argument"));
            }
            let lit: LitStr = meta.value()?.parse()?;
            raw_args.cache = Some(lit.value());
            Ok(())
        } else {
            Err(meta.error("unsupported attribute; expected `cache = \"...\"`"))
        }
    });

    // 2. Use the `with` syntax to parse the TokenStream using our parser closure
    parse_macro_input!(args with args_parser);
    let input_fn = parse_macro_input!(input as ItemFn);

    match loader_impl(raw_args, input_fn) {
        Ok(ts) => ts,
        Err(err) => err.to_compile_error().into(),
    }
}

fn loader_impl(raw_args: LoaderArgs, input_fn: ItemFn) -> Result<TokenStream> {
    let original_name = input_fn.sig.ident.clone();
    let original_name_str = original_name.to_string();
    let args = finalize_args(raw_args, &original_name_str);

    let uncached_name = format_ident!("{}_uncached", original_name);
    let mut uncached_fn = input_fn.clone();
    uncached_fn.sig.ident = uncached_name.clone();

    uncached_fn
        .attrs
        .retain(|attr| !attr.path().is_ident("loader"));

    let mut wrapper_sig = input_fn.sig.clone();
    wrapper_sig.ident = original_name.clone();

    let mut arg_idents = Vec::new();
    for arg in &wrapper_sig.inputs {
        match arg {
            FnArg::Receiver(_) => {
                return Err(syn::Error::new_spanned(
                    arg,
                    "loader can only be used on free functions",
                ));
            }
            FnArg::Typed(pat_ty) => match &*pat_ty.pat {
                Pat::Ident(pat_ident) => arg_idents.push(pat_ident.ident.clone()),
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "all parameters must be simple identifiers",
                    ));
                }
            },
        }
    }

    let cache_ty = format_ident!("P2");
    wrapper_sig
        .generics
        .params
        .push(syn::parse_quote!(#cache_ty));
    wrapper_sig
        .inputs
        .push(syn::parse_quote!(cache_path: &#cache_ty));
    wrapper_sig
        .generics
        .where_clause
        .get_or_insert_with(|| syn::WhereClause {
            where_token: Default::default(),
            predicates: Default::default(),
        })
        .predicates
        .push(syn::parse_quote!(P2: AsRef<Path> + ?Sized));

    let cache_file_name = args.cache;

    let wrapper_block = quote!({
        if !cache_path.as_ref().exists() {
            log::warn!(
                "Cache dir '{}' does not exist. Falling back to uncached loading.",
                cache_path.as_ref().display()
            );
            log::info!("Loading hypergraph from source");
            let rv = #uncached_name(#(#arg_idents),*)?;
            return Ok(rv);
        }

        let cache_file = cache_path.as_ref().join(#cache_file_name);
        if cache_file.exists() {
            match Hypergraph::load_deserialized(&cache_file) {
                Ok(hg) => Ok(hg),
                Err(_err) => {
                    log::warn!(
                        "Cache file {} is corrupted. Falling back to uncached loading.",
                        cache_file.display()
                    );
                    let rv = #uncached_name(#(#arg_idents),*)?;
                    rv.save_to_file(&cache_file)?;
                    Ok(rv)
                }
            }
        } else {
            log::info!(
                "Loading hypergraph from source and caching to {}...",
                cache_file.display()
            );
            let rv = #uncached_name(#(#arg_idents),*)?;
            rv.save_to_file(&cache_file)?;
            Ok(rv)
        }
    });

    let wrapper_fn = syn::ItemFn {
        attrs: input_fn.attrs.clone(),
        vis: input_fn.vis.clone(),
        sig: wrapper_sig,
        block: Box::new(syn::parse_quote!(#wrapper_block)),
    };

    Ok(quote! {
        #uncached_fn
        #wrapper_fn
    }
    .into())
}
