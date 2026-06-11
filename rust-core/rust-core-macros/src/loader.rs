use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::ToTokens;
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{Field, Ident, Item, ItemMod, Visibility, parse_macro_input};

// Helper struct to keep track of parsed loader metadata
struct StructInfo {
    ident: Ident,
    vis: Visibility,
    attrs: Vec<syn::Attribute>,
    impl_attrs: Vec<syn::Attribute>, // <-- Added to track forwarded attributes
    parent: Option<Ident>,
    local_fields: Vec<Field>,
    children: Vec<Ident>,
}

pub fn loaders(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut item_mod = parse_macro_input!(item as ItemMod);

    let Some((brace, items)) = item_mod.content.take() else {
        return item_mod.into_token_stream().into();
    };

    let mut managed_structs = HashMap::new();
    let mut other_items = Vec::new();

    // 1. First pass: Collect loader/subloader structs and isolate regular items
    for item in items {
        if let Item::Struct(mut strct) = item {
            let mut is_loader = false;
            let mut parent = None;
            let mut retained_attrs = Vec::new();
            let mut impl_attrs = Vec::new(); // <-- Temporary container for this struct's impl attributes

            // Check attributes and extract configurations
            for attr in strct.attrs.iter() {
                if attr.path().is_ident("loader") {
                    is_loader = true;
                } else if attr.path().is_ident("subloader") {
                    is_loader = true;
                    if let Ok(parent_ident) = attr.parse_args::<Ident>() {
                        parent = Some(parent_ident);
                    }
                } else if attr.path().is_ident("impl_attr") {
                    // Extract the contents of #[impl_attr(...)] and turn them into a real attribute
                    if let syn::Meta::List(meta_list) = &attr.meta {
                        let inner_tokens = &meta_list.tokens;
                        let forwarded_attr: syn::Attribute = syn::parse_quote! { #[#inner_tokens] };
                        impl_attrs.push(forwarded_attr);
                    }
                } else {
                    retained_attrs.push(attr.clone());
                }
            }

            if is_loader {
                strct.attrs = retained_attrs; // Strip custom helper attributes from the final struct
                let local_fields = match strct.fields {
                    syn::Fields::Named(fields) => fields.named.into_iter().collect(),
                    _ => Vec::new(),
                };

                managed_structs.insert(
                    strct.ident.clone(),
                    StructInfo {
                        ident: strct.ident,
                        vis: strct.vis,
                        attrs: strct.attrs,
                        impl_attrs, // <-- Store them here
                        parent,
                        local_fields,
                        children: Vec::new(),
                    },
                );
            } else {
                other_items.push(Item::Struct(strct));
            }
        } else {
            other_items.push(item);
        }
    }

    // 2. Map out parent-to-child relationships
    let keys: Vec<Ident> = managed_structs.keys().cloned().collect();
    for key in &keys {
        if let Some(parent_ident) = managed_structs.get(key).unwrap().parent.clone() {
            if let Some(parent_info) = managed_structs.get_mut(&parent_ident) {
                parent_info.children.push(key.clone());
            }
        }
    }

    // 3. Recursive helper to resolve field inheritance across the hierarchy
    fn resolve_all_fields(
        ident: &Ident,
        structs: &HashMap<Ident, StructInfo>,
        cache: &mut HashMap<Ident, Vec<Field>>,
    ) -> Vec<Field> {
        if let Some(fields) = cache.get(ident) {
            return fields.clone();
        }

        let info = &structs[ident];
        let mut all_fields = Vec::new();

        if let Some(ref parent_ident) = info.parent {
            if structs.contains_key(parent_ident) {
                all_fields.extend(resolve_all_fields(parent_ident, structs, cache));
            }
        }

        all_fields.extend(info.local_fields.clone());
        cache.insert(ident.clone(), all_fields.clone());
        all_fields
    }

    let mut fields_cache = HashMap::new();
    let mut generated_items = Vec::<TokenStream2>::new();

    // 4. Code Generation Phase
    for info in managed_structs.values() {
        let struct_ident = &info.ident;
        let struct_vis = &info.vis;
        let struct_attrs = &info.attrs;
        let impl_attrs = &info.impl_attrs;

        let all_fields = resolve_all_fields(struct_ident, &managed_structs, &mut fields_cache);

        // Generate Struct Definition
        let fields_tokens = all_fields.iter().map(|f| {
            let f_vis = &f.vis;
            let f_ident = &f.ident;
            let f_ty = &f.ty;

            let f_attrs = f.attrs.iter().filter(|attr| {
                !attr.path().is_ident("builder")
                    && !attr
                        .path()
                        .segments
                        .iter()
                        .any(|seg| seg.ident == "builder")
            });

            quote! { #(#f_attrs)* #f_vis #f_ident: #f_ty }
        });

        generated_items.push(quote! {
            #[derive(Default, Clone)]
            #[pyclass]
            #(#struct_attrs)*
            #struct_vis struct #struct_ident {
                #(#fields_tokens,)*
            }
        });

        // Generate Builder Methods
        let builder_methods = all_fields.iter().filter_map(|f| {
            let f_ident = &f.ident;
            let f_ty = &f.ty;

            let should_skip = f.attrs.iter().any(|attr| {
                attr.path().segments.len() == 2
                    && attr.path().segments[0].ident == "builder"
                    && attr.path().segments[1].ident == "skip"
            });

            if should_skip {
                None
            } else {
                Some(quote! {
                    pub fn #f_ident(&mut self, #f_ident: #f_ty) -> Self {
                        self.#f_ident = #f_ident;
                        self.clone()
                    }
                })
            }
        });

        // Generate Transitions or Leaf Actions
        let context_methods = if !info.children.is_empty() {
            let transitions = info.children.iter().map(|child_ident| {
                let method_name = format_ident!("{}", to_snake_case(&child_ident.to_string()));
                let copy_fields = all_fields.iter().map(|f| {
                    let f_ident = &f.ident;
                    quote! { rv.#f_ident = self.#f_ident.clone(); }
                });

                quote! {
                    pub fn #method_name(&self) -> #child_ident {
                        let mut rv = #child_ident::default();
                        #(#copy_fields)*
                        rv
                    }
                }
            });
            quote! { #(#transitions)* }
        } else {
            quote! {
            pub fn load(&self) -> PyResult<<Self as Loader>::Output> {
                    let dataset_location = self.dataset_location.clone();
                    <Self as Loader>::load(self).map_err(|e| PyIOError::new_err(format!("Could not load {}", dataset_location.display())))
                }
            }
        };

        // Combine into structure impl block
        generated_items.push(quote! {
            #(#impl_attrs)*
            #[pymethods]
            impl #struct_ident {
                #[staticmethod]
                pub fn builder() -> Self {
                    Self::default()
                }

                #(#builder_methods)*
                #context_methods
            }
        });

        // Map fields to hash statements. This assumes all field types implement std::hash::Hash.
        let hash_fields = all_fields.iter().map(|f| {
            let f_ident = &f.ident;
            quote! { std::hash::Hash::hash(&self.#f_ident, &mut state); }
        });

        generated_items.push(quote! {
            impl DatasetInfo for #struct_ident {
                fn dataset_location(&self) -> std::path::PathBuf {
                    self.dataset_location.clone()
                }

                fn cache_dir(&self) -> Option<std::path::PathBuf> {
                    self.cache_dir.clone()
                }

                fn cache_hash(&self, length: usize) -> String {
                    use std::hash::Hasher;
                    let mut state = std::collections::hash_map::DefaultHasher::new();

                    // Feed all available fields dynamically into the hasher
                    #(#hash_fields)*

                    let hash_value = state.finish();
                    let hex_string = format!("{:x}", hash_value);

                    // Truncate to the requested length safely
                    if length >= hex_string.len() {
                        hex_string
                    } else {
                        hex_string[..length].to_string()
                    }
                }
            }
        });
    }

    let final_tokens = quote! {
        #(#other_items)*
        #(#generated_items)*
    };

    item_mod.content = Some((brace, vec![Item::Verbatim(final_tokens)]));
    item_mod.into_token_stream().into()
}

fn to_snake_case(s: &str) -> String {
    let mut snake = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                snake.push('_');
            }
            snake.push(ch.to_ascii_lowercase());
        } else {
            snake.push(ch);
        }
    }
    snake.replace("_d_b", "_db")
}
