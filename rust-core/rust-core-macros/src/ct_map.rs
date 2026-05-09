use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, quote};
use syn::{
    Expr, ExprRange, Ident, ItemStruct, Type, parenthesized,
    parse::{Parse, ParseStream},
    spanned::Spanned,
};

#[derive(Clone, Copy, Debug, Default)]
struct GenFlags {
    take: bool,
    get: bool,
    get_mut: bool,
    new: bool,
}

impl GenFlags {
    fn all() -> Self {
        Self {
            take: true,
            get: true,
            get_mut: true,
            new: true,
        }
    }

    fn none() -> Self {
        Self::default()
    }

    fn disable(&mut self, other: GenFlags) {
        if other.take {
            self.take = false;
        }
        if other.get {
            self.get = false;
        }
        if other.get_mut {
            self.get_mut = false;
        }
        if other.new {
            self.new = false;
        }
    }
}

struct Args {
    ty: Type,
    range: ExprRange,
    default_expr: Option<Expr>,
    generate: Option<GenFlags>,
    skip: Option<GenFlags>,
}

struct ArgsFinalized {
    ty: Type,
    range_start: usize,
    range_end: usize,
    default_expr: Expr,
    gen_flags: GenFlags,
}

impl Parse for Args {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut ty = None;
        let mut range = None;
        let mut default_expr = None;
        let mut generate = None;
        let mut skip = None;

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            let content;
            parenthesized!(content in input);

            match ident.to_string().as_str() {
                "ty" => ty = Some(content.parse()?),
                "rg" => range = Some(content.parse()?),
                "default" => default_expr = Some(content.parse()?),
                "gen" => {
                    if skip.is_some() {
                        return Err(syn::Error::new(
                            ident.span(),
                            "cannot specify both gen(...) and skip(...)",
                        ));
                    }
                    generate = Some(parse_flags(&content)?);
                }
                "skip" => {
                    if generate.is_some() {
                        return Err(syn::Error::new(
                            ident.span(),
                            "cannot specify both gen(...) and skip(...)",
                        ));
                    }
                    skip = Some(parse_flags(&content)?);
                }
                _ => return Err(syn::Error::new(ident.span(), "unknown ct_map option")),
            }

            if input.peek(syn::Token![,]) {
                input.parse::<syn::Token![,]>()?;
            }
        }

        Ok(Self {
            ty: ty.ok_or_else(|| syn::Error::new(Span::call_site(), "missing ty(...)"))?,
            range: range.ok_or_else(|| syn::Error::new(Span::call_site(), "missing rg(...)"))?,
            default_expr,
            generate,
            skip,
        })
    }
}

impl Args {
    fn finalize(self) -> syn::Result<ArgsFinalized> {
        let range_expr = self.range;

        let start_expr = range_expr
            .clone()
            .start
            .ok_or_else(|| syn::Error::new(range_expr.span(), "range must have start"))?;

        let end_expr = range_expr
            .clone()
            .end
            .ok_or_else(|| syn::Error::new(range_expr.span(), "range must have end"))?;

        let range_start = expr_to_usize(&start_expr)?;
        let range_end = expr_to_usize(&end_expr)?;

        if range_end <= range_start {
            return Err(syn::Error::new(
                Span::call_site(),
                "range end must be > range start",
            ));
        }

        let gen_flags = if let Some(generate) = self.generate {
            generate
        } else if let Some(skip) = self.skip {
            let mut flags = GenFlags::all();
            flags.disable(skip);
            flags
        } else {
            GenFlags::all()
        };

        let ty = self.ty;
        let default_expr = self.default_expr.unwrap_or_else(|| {
            syn::parse_quote! { <#ty as ::core::default::Default>::default() }
        });

        Ok(ArgsFinalized {
            ty,
            range_start,
            range_end,
            default_expr,
            gen_flags,
        })
    }
}

fn expr_to_usize(expr: &Expr) -> syn::Result<usize> {
    match expr {
        Expr::Lit(expr_lit) => match &expr_lit.lit {
            syn::Lit::Int(lit) => lit.base10_parse(),
            _ => Err(syn::Error::new(expr.span(), "expected integer literal")),
        },
        _ => Err(syn::Error::new(expr.span(), "expected integer literal")),
    }
}

fn parse_flags(input: ParseStream) -> syn::Result<GenFlags> {
    let mut flags = GenFlags::none();

    while !input.is_empty() {
        let ident: Ident = input.parse()?;
        match ident.to_string().as_str() {
            "take" => flags.take = true,
            "get" => flags.get = true,
            "get_mut" => flags.get_mut = true,
            "new" => flags.new = true,
            _ => return Err(syn::Error::new(ident.span(), "unknown generation flag")),
        }

        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
    }

    Ok(flags)
}

fn replace_idents(stream: TokenStream, n_val: i64, i_val: i64) -> TokenStream {
    use proc_macro2::{Group, Literal, TokenTree};

    stream
        .into_iter()
        .map(|tt| match tt {
            TokenTree::Ident(ref ident) if ident == "N" => {
                TokenTree::Literal(Literal::i64_unsuffixed(n_val))
            }
            TokenTree::Ident(ref ident) if ident == "I" => {
                TokenTree::Literal(Literal::i64_unsuffixed(i_val))
            }
            TokenTree::Group(group) => {
                let delim = group.delimiter();
                let span = group.span();
                let inner = replace_idents(group.stream(), n_val, i_val);
                let mut new_group = Group::new(delim, inner);
                new_group.set_span(span);
                TokenTree::Group(new_group)
            }
            other => other,
        })
        .collect()
}

fn expand_ct_map(args: ArgsFinalized, input: ItemStruct) -> TokenStream {
    let struct_name = input.ident;
    let min = args.range_start;
    let max = args.range_end;
    let count = max - min;

    let ty_stream = args.ty.to_token_stream();
    let default_stream = args.default_expr.to_token_stream();
    let flags = args.gen_flags;

    let fields = (min..max).enumerate().map(|(i, n)| {
        let specialized_type = replace_idents(ty_stream.clone(), n as i64, i as i64);
        quote! { #specialized_type }
    });

    let get_immutable = if flags.get {
        let methods = (min..max).enumerate().map(|(i, n)| {
            let method_name = quote::format_ident!("get_{}", n);
            let specialized_type = replace_idents(ty_stream.clone(), n as i64, i as i64);
            let tuple_index = syn::Index::from(i);
            quote! {
                pub fn #method_name(&self) -> &#specialized_type {
                    &self.#tuple_index
                }
            }
        });
        quote! { #(#methods)* }
    } else {
        quote! {}
    };

    let get_mutable = if flags.get_mut {
        let methods = (min..max).enumerate().map(|(i, n)| {
            let method_name = quote::format_ident!("get_{}_mut", n);
            let specialized_type = replace_idents(ty_stream.clone(), n as i64, i as i64);
            let tuple_index = syn::Index::from(i);
            quote! {
                pub fn #method_name(&mut self) -> &mut #specialized_type {
                    &mut self.#tuple_index
                }
            }
        });
        quote! { #(#methods)* }
    } else {
        quote! {}
    };

    let take = if flags.take {
        let methods = (min..max).enumerate().map(|(i, n)| {
            let method_name = quote::format_ident!("take_{}", n);
            let specialized_type = replace_idents(ty_stream.clone(), n as i64, i as i64);
            let tuple_index = syn::Index::from(i);
            quote! {
                pub fn #method_name(&mut self) -> #specialized_type {
                    ::core::mem::take(&mut self.#tuple_index)
                }
            }
        });
        quote! { #(#methods)* }
    } else {
        quote! {}
    };

    let new_fn = if flags.new {
        let defaults = (min..max)
            .enumerate()
            .map(|(i, n)| replace_idents(default_stream.clone(), n as i64, i as i64));

        quote! {
            pub fn new() -> Self {
                Self(
                    #(#defaults),*
                )
            }
        }
    } else {
        quote! {}
    };

    quote! {
        pub struct #struct_name(
            #(#fields),*
        );

        impl #struct_name {
            pub const SIZE: usize = #count;

            #new_fn
            #get_immutable
            #get_mutable
            #take
        }
    }
}

pub fn ct_map(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let args = syn::parse_macro_input!(attr as Args);

    let finalized = match args.finalize() {
        Ok(v) => v,
        Err(e) => return e.into_compile_error().into(),
    };

    let input_struct = syn::parse_macro_input!(item as ItemStruct);
    expand_ct_map(finalized, input_struct).into()
}
