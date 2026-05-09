use proc_macro::TokenStream;
use quote::quote;
use syn::{
    FnArg, ImplItem, ItemImpl, Meta, Pat, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

struct Args {
    extra_attrs: Vec<syn::Attribute>,
}

impl Parse for Args {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse comma-separated Meta items: attr(pymethods), other(thing)
        let vars = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;
        let mut extra_attrs = Vec::new();

        for meta in vars {
            if meta.path().is_ident("attr") {
                // Extract what's inside the parentheses of attr(...)
                if let Meta::List(list) = meta {
                    let inner_tokens = &list.tokens;
                    extra_attrs.push(syn::parse_quote!(#[#inner_tokens]));
                }
            }
        }
        Ok(Args { extra_attrs })
    }
}

pub fn inherent(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as Args);

    let mut input = parse_macro_input!(item as ItemImpl);
    let self_ty = &input.self_ty;

    let mut inherent_methods = Vec::new();
    let trait_path = if let Some((_, path, _)) = &input.trait_ {
        quote! { #path }
    } else {
        quote! { #self_ty } // Fallback
    };

    for item in &mut input.items {
        if let ImplItem::Fn(method) = item {
            let vis = match &method.vis {
                syn::Visibility::Inherited => quote! { pub },
                other => quote! { #other },
            };

            let sig = &method.sig;
            let ident = &sig.ident;

            let args_tokens = sig.inputs.iter().map(|arg| match arg {
                FnArg::Receiver(_) => quote! { self },
                FnArg::Typed(pat_type) => {
                    if let Pat::Ident(pat_id) = &*pat_type.pat {
                        let id = &pat_id.ident;
                        quote! { #id }
                    } else {
                        quote! { _ }
                    }
                }
            });

            inherent_methods.push(quote! {
                #vis #sig {
                    <Self as #trait_path>::#ident(#(#args_tokens),*)
                }
            });

            method.vis = syn::Visibility::Inherited;
        }
    }

    let extra_attrs = &args.extra_attrs;

    let expanded = quote! {
        #input

        #(#extra_attrs)*
        impl #self_ty {
            #(#inherent_methods)*
        }
    };

    TokenStream::from(expanded)
}
